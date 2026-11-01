"""Discretisations of Ducceschi et al. (2021), Sections II–VI."""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


@dataclass
class Model:
    mass: np.ndarray
    stiffness: sparse.csr_matrix
    contact: sparse.csr_matrix
    barrier: np.ndarray
    kappa: np.ndarray
    exponent: np.ndarray
    q: np.ndarray
    previous: np.ndarray
    pickup: np.ndarray
    rate: int = 44100
    smoothing: int = 0
    damping: np.ndarray = None
    loss: sparse.csr_matrix = None
    bilateral: np.ndarray = None
    rule: np.ndarray = None
    psi: np.ndarray = None
    bow: np.ndarray = None
    bow_force: float = 0
    bow_speed: float = 0

    def __post_init__(self):
        for names, size in [(('damping', 'bow'), len(self.mass)), (('bilateral', 'rule', 'psi'), len(self.barrier))]:
            for name in names:
                if getattr(self, name) is None:
                    setattr(self, name, np.zeros(size))
        if self.loss is None:
            self.loss = sparse.csr_matrix((len(self.mass), len(self.mass)))

    def write(self, path):
        n, c = len(self.mass), len(self.barrier)
        dt = 1 / self.rate
        matrices = [self.stiffness.tocsr(), self.loss.tocsr(), self.contact.tocsr()]
        for matrix in matrices:
            matrix.sum_duplicates()
            matrix.eliminate_zeros()
            matrix.sort_indices()
        # The staggered energy is positive when M - dt R/2 - dt² K/4 is positive.
        inverse_mass = sparse.diags(1 / np.sqrt(self.mass))
        stability = inverse_mass @ (dt * self.loss / 2 + dt**2 * self.stiffness / 4) @ inverse_mass
        bound = float(eigsh(stability, k=1, which='LA', v0=np.random.default_rng(0).standard_normal(n), return_eigenvectors=False)[0]) if n > 1 else float(stability[0, 0])
        if bound >= 1:
            raise ValueError(f'Unstable linear discretisation: {bound}')
        gram = (self.contact @ sparse.diags(dt**2 / (self.mass + dt*self.damping/2)) @ self.contact.T).toarray()
        coefficients = np.concatenate([self.mass, self.damping, self.pickup, self.bow,
                                       self.barrier, self.kappa, self.exponent, self.bilateral, self.rule,
                                       *(m.data for m in matrices), gram.ravel()])
        indices = np.concatenate([a for m in matrices for a in (m.indptr, m.indices)])
        initial = np.concatenate([self.q, self.previous, self.psi])
        with path.open('w') as f:
            f.write(f'{n} {c} {matrices[0].nnz} {matrices[1].nnz} {matrices[2].nnz} {self.smoothing} {dt:.17g} {self.bow_force:.17g} {self.bow_speed:.17g} 100\n')
            np.savetxt(f, coefficients, fmt='%.17g')
            np.savetxt(f, indices, fmt='%d')
            np.savetxt(f, initial, fmt='%.17g')
        response = float(np.sum(self.bow**2/(self.mass+dt*self.damping/2))*dt/2)
        return {'dofs': n, 'contacts': c, 'linear_stability_ratio': bound, 'sample_rate': self.rate,
                'bow_residual_derivative_lower_bound': 1-response*self.bow_force*2*np.sqrt(200)/np.e}


def laplacian(n, h):
    return sparse.diags([np.ones(n-1), -2*np.ones(n), np.ones(n-1)], [-1, 0, 1], format='csr') / h**2


def interpolate(positions, intervals, length):
    rows, cols, values = [], [], []
    for row, position in enumerate(np.atleast_1d(positions)):
        x = position * intervals / length
        lower = int(np.floor(x))
        for node, value in ((lower, 1 - (x-lower)), (lower+1, x-lower)):
            if 0 < node < intervals and value != 0:
                rows.append(row); cols.append(node-1); values.append(value)
    return sparse.csr_matrix((values, (rows, cols)), shape=(len(np.atleast_1d(positions)), intervals-1))


def mass_barrier(rate=44100, stiffness=1e13, exponent=1., frequency=0, velocity=1.5):
    dt = 1/rate
    x = -velocity*(np.floor(.01/velocity*rate)-.5)*dt
    omega2 = (2*np.pi*frequency)**2
    return Model(np.array([.01]), sparse.csr_matrix([[.01*omega2]]), sparse.eye(1, format='csr'), np.zeros(1),
                 np.array([stiffness]), np.array([exponent]), np.array([x+velocity*dt-.5*dt**2*omega2*x]), np.array([x]), np.ones(1), rate=rate, rule=np.ones(1))


def string_model(kind='hammer', paper=False, modal=False, rate=44100, stiffness=None):
    length = .7
    density = .0063 if paper else 8000*np.pi*(.0005 if kind == 'hammer' else .0003)**2
    tension = 100 if paper else (10 if kind == 'hammer' else 30)
    intervals = int(length*rate/np.sqrt(tension/density))
    h = length/intervals
    x = np.arange(1, intervals)*h
    if kind == 'hammer':
        contact = sparse.hstack([-interpolate([.23*length], intervals, length), sparse.csr_matrix([[1.]])], format='csr')
        mass = np.r_[np.full(intervals-1, density*h), .01]
        k = sparse.block_diag([-tension*h*laplacian(intervals-1, h), sparse.csr_matrix((1, 1))], format='csr')
        previous = np.r_[np.zeros(intervals-1), -.001]
        q = previous.copy(); q[-1] += (.5 if paper else .2)/rate
        pickup = np.r_[interpolate([.68*length], intervals, length).toarray()[0], 0.]
        return Model(mass, k, contact, np.zeros(1), np.array([stiffness or (1e12 if paper else 1e7)]), np.array([1.3]), q, previous, pickup, rate=rate, smoothing=1, rule=np.full(1, 2.))
    positions = x if kind == 'backboard' else length-length*2.**(-np.arange(1, 13)/12)
    height = -.0001 if paper else -.001
    barrier = height-.001*positions-.02*positions**2 if kind == 'backboard' else np.full(len(positions), height)
    if modal:
        modes = np.arange(1, int(2*length*rate/np.pi*np.sqrt(density/tension))+1)
        mass = np.full(len(modes), density*length/2)
        k = sparse.diags(tension*length/2*(modes*np.pi/length)**2, format='csr')
        contact = sparse.csr_matrix(-np.sin(np.outer(positions, modes)*np.pi/length))
        q = np.zeros(len(modes)); q[0] = .003
        pickup = np.sin(modes*np.pi*.68)
    else:
        mass = np.full(len(x), density*h)
        k = -tension*h*laplacian(len(x), h)
        contact = -interpolate(positions, intervals, length)
        q = .003*np.sin(np.pi*x/length)
        pickup = interpolate([.68*length], intervals, length).toarray()[0]
    return Model(mass, k, contact, barrier, np.full(len(positions), stiffness or (1e12 if paper else 1e13)), np.full(len(positions), 1.2), q, q.copy(), pickup, rate=rate, smoothing=2)


def cubic(position, intervals, length):
    x = position*intervals/length
    node = int(np.floor(x)); a = x-node
    weights = [-a*(a-1)*(a-2)/6, (a-1)*(a+1)*(a-2)/2, -a*(a+1)*(a-2)/2, a*(a+1)*(a-1)/6]
    return sparse.csr_matrix((weights, ([0]*4, np.arange(node-2, node+2))), shape=(1, intervals-1))


def point(position, intervals, length):
    node = int(np.floor(position*intervals/length+.5))
    return sparse.csr_matrix(([1.], ([0], [node-1])), shape=(1, intervals-1))


def snare(rate=44100):
    radius, density, tension = .15, .2, 2000.
    h = np.sqrt(2*tension/density)/rate * 1.01
    axis = np.arange(-int(radius/h), int(radius/h)+1)*h
    xx, yy = np.meshgrid(axis, axis, indexing='ij')
    inside = xx**2+yy**2 < radius**2
    ids = np.full(inside.shape, -1); ids[inside] = np.arange(inside.sum())
    rows, cols, values = [], [], []
    for i, j in zip(*np.where(inside)):
        index = ids[i, j]
        rows.append(index); cols.append(index); values.append(-4/h**2)
        for a, b in ((i-1, j), (i+1, j), (i, j-1), (i, j+1)):
            if 0 <= a < len(axis) and 0 <= b < len(axis) and inside[a, b]:
                rows.append(index); cols.append(ids[a, b]); values.append(1/h**2)
    membrane = sparse.csr_matrix((values, (rows, cols)), shape=(inside.sum(), inside.sum()))
    def membrane_pickup(points):
        rows, cols, values = [], [], []
        for row, (x, y) in enumerate(points):
            sx, sy = (x-axis[0])/h, (y-axis[0])/h
            ix, iy = int(np.floor(sx)), int(np.floor(sy))
            for dx, wx in ((0, 1-sx+ix), (1, sx-ix)):
                for dy, wy in ((0, 1-sy+iy), (1, sy-iy)):
                    node = ids[ix+dx, iy+dy]
                    if node < 0: raise ValueError('Wire interpolation outside membrane')
                    rows.append(row); cols.append(node); values.append(wx*wy)
        return sparse.csr_matrix((values, (rows, cols)), shape=(len(points), inside.sum()))
    length, wire_density, wire_tension = .2, .001, 10.
    intervals = int(length*rate/np.sqrt(wire_tension/wire_density))
    hw = length/intervals
    x = np.arange(1, intervals)*hw
    # Figure 7 prints x=0.3 m outside its R=0.15 m disk; x=0.03 m is used explicitly here.
    interpolation = membrane_pickup(np.c_[np.full(len(x), .03), length/2-x])
    contact = sparse.hstack([interpolation, -sparse.eye(len(x))], format='csr')
    mass = np.r_[np.full(inside.sum(), density*h*h), np.full(len(x), wire_density*hw)]
    k = sparse.block_diag([-tension*h*h*membrane, -wire_tension*hw*laplacian(len(x), hw)], format='csr')
    q = np.r_[np.zeros(inside.sum()), .001*np.sin(np.pi*x/length)]
    pickup = np.r_[membrane_pickup([(.06, .02)]).toarray()[0], np.zeros(len(x))]
    return Model(mass, k, contact, np.full(len(x), -.0004), np.full(len(x), 1e12*hw), np.full(len(x), 1.3), q, q.copy(), pickup, rate=rate, smoothing=1, rule=np.full(len(x), 2.))


def tromba(rate=44100, bow_force=.1):
    dt = 1/rate
    length, density, radius, young = 1.9, 7850., .0005, 2e11
    linear = density*np.pi*radius**2
    tension = linear*(2*length*32)**2
    rigidity = young*np.pi*radius**4/4
    sigma0, sigma1 = .1, .05
    c2, k2 = tension/linear, rigidity/linear
    hmin = np.sqrt((c2*dt**2+4*sigma1*dt+np.sqrt((c2*dt**2+4*sigma1*dt)**2+16*k2*dt**2))/2)
    ns = int(length/hmin); hs = length/ns
    ds = laplacian(ns-1, hs)
    ks = hs*(-tension*ds+rigidity*(ds@ds))
    # The predecessor uses a 1 cm minimum plate spacing for its real-time implementation.
    lx, ly, surface_density = .18, 1.35, 50*.01
    rigidity_plate = 2e5*.01**3/(12*(1-.3**2))
    hp = max(.01, 2*np.sqrt(dt*(.05+np.sqrt(.05**2+rigidity_plate/surface_density))))
    nx, ny = int(lx/hp), int(ly/hp)
    hx, hy = lx/nx, ly/ny
    dp = sparse.kronsum(laplacian(ny-1, hy), laplacian(nx-1, hx), format='csr')
    area = hx*hy
    edge = np.zeros((nx-1, ny-1))
    edge[0, :] += 2/hx**4; edge[-1, :] += 2/hx**4
    edge[:, 0] += 2/hy**4; edge[:, -1] += 2/hy**4
    kp = rigidity_plate*area*(dp@dp+sparse.diags(edge.ravel()))
    pcontact = sparse.kron(point(.135, nx, lx), point(1.08, ny, ly), format='csr')
    scontact = point(1.65, ns, length)
    zs = sparse.csr_matrix((1, ns-1)); zp = sparse.csr_matrix((1, dp.shape[0]))
    contact = sparse.vstack([sparse.hstack([zs, pcontact, sparse.csr_matrix([[-1.]])]), sparse.hstack([-scontact, zp, sparse.csr_matrix([[1.]])])], format='csr')
    mass = np.r_[np.full(ns-1, linear*hs), np.full(dp.shape[0], surface_density*area), .001]
    k = sparse.block_diag([ks, kp, sparse.csr_matrix([[.001*(2*np.pi*500)**2]])], format='csr')
    loss = sparse.block_diag([-2*linear*hs*sigma1*ds, -2*surface_density*area*.05*dp, sparse.csr_matrix((1, 1))], format='csr')
    damping = mass*np.r_[np.full(ns-1, 2*sigma0), np.full(dp.shape[0], 4.), .05]
    q = np.zeros(len(mass))
    pickup = np.r_[.1*interpolate([length-1.65], ns, length).toarray()[0], pcontact.toarray()[0], 0.]
    bow = np.r_[cubic(length/3, ns, length).toarray()[0], np.zeros(dp.shape[0]+1)]
    return Model(mass, k, contact, np.array([-5e-6, 0.]), np.array([5e8, 5e6]), np.full(2, 1.3), q, q.copy(), pickup, rate=rate, smoothing=1,
                 damping=damping, loss=loss, bilateral=np.array([0., 1.]), rule=np.array([2., 0.]), bow=bow, bow_force=bow_force, bow_speed=-.2)
