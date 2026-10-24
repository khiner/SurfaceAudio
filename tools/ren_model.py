"""Ren 2010 surface preparation and its Raghuvanshi–Lin 2006 spring-mass resonator."""
import numpy as np
from scipy.linalg import eigh


def spring_system(vertices, triangles, young, density, thickness):
    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles)
            or not np.issubdtype(triangles.dtype, np.integer) or triangles.min() < 0
            or triangles.max() >= len(vertices) or not np.isfinite([young, density, thickness]).all()
            or min(young, density, thickness) <= 0):
        raise ValueError('Invalid spring mesh or material')
    corners = vertices[triangles]
    areas = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1) / 2
    if np.any(areas <= 0):
        raise ValueError('Degenerate mesh face')
    masses = np.bincount(triangles.ravel(), weights=np.repeat(areas * density * thickness / 3, 3), minlength=len(vertices))
    if np.any(masses <= 0):
        raise ValueError('Unconnected mesh vertex')
    edges = np.unique(np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1), axis=0)
    stiffness = np.zeros((3 * len(vertices), 3 * len(vertices)))
    for i, j in edges:
        direction = vertices[j] - vertices[i]
        direction /= np.linalg.norm(direction)
        spring = young * thickness * np.outer(direction, direction)
        for a, b, sign in ((i, i, 1), (j, j, 1), (i, j, -1), (j, i, -1)):
            stiffness[3*a:3*a+3, 3*b:3*b+3] += sign * spring
    return np.repeat(masses, 3), stiffness


def mesh_modes(vertices, triangles, young, density, thickness, gamma, eta, pickup, rate=44100):
    mass, stiffness = spring_system(vertices, triangles, young, density, thickness)
    if not np.isfinite([gamma, eta, rate]).all() or gamma <= 0 or eta < 0 or rate <= 0:
        raise ValueError('Invalid Rayleigh damping or sample rate')
    pickup = np.asarray(pickup, dtype=float)
    if pickup.shape != mass.shape or not np.isfinite(pickup).all():
        raise ValueError('Expected one pickup weight per displacement coordinate')
    eigenvalues, shapes = eigh(stiffness, np.diag(mass))
    decay_rate = (gamma + eta * eigenvalues) / 2
    omega2 = eigenvalues - decay_rate**2
    keep = (omega2 > (2*np.pi*20)**2) & (omega2 < (np.pi*rate)**2)
    omega = np.sqrt(omega2[keep])
    shapes = shapes[:, keep]
    if not len(omega):
        raise ValueError('Mesh has no audible underdamped modes')
    modes = np.column_stack([omega/(2*np.pi), 1/decay_rate[keep], np.einsum('i,ij->j', pickup, shapes) / omega])
    elastic = np.einsum('ij,jk->ik', stiffness, shapes)
    residual = elastic - mass[:, None] * shapes * eigenvalues[keep]
    diagnostics = {'mode_count': len(omega), 'omitted_modes': int(len(eigenvalues) - len(omega)),
                   'eigen_relative_residual': float(np.linalg.norm(residual) / np.linalg.norm(elastic)),
                   'mass_orthogonality_error': float(np.max(np.abs(np.einsum('ki,k,kj->ij', shapes, mass, shapes) - np.eye(len(omega)))))}
    return modes, shapes, diagnostics


def normal_map(heights, extent):
    """Periodic centered height derivatives, with rows along V and columns along U."""
    heights = np.asarray(heights, dtype=float)
    extent = np.asarray(extent, dtype=float)
    if (heights.ndim != 2 or min(heights.shape) < 3 or not np.isfinite(heights).all()
            or extent.shape != (2,) or not np.isfinite(extent).all() or np.any(extent <= 0)):
        raise ValueError('Invalid periodic height map')
    dx, dy = extent / heights.shape[::-1]
    x = (np.roll(heights, -1, 1) - np.roll(heights, 1, 1)) / (2*dx)
    y = (np.roll(heights, -1, 0) - np.roll(heights, 1, 0)) / (2*dy)
    normals = np.stack([-x, -y, np.ones_like(x)], axis=-1)
    return normals / np.linalg.norm(normals, axis=-1, keepdims=True)


def fractal_table(count, exponent, seed, cutoff=1.):
    """Unit-RMS periodic noise with power proportional to spatial frequency**(-exponent)."""
    if count < 4 or count % 2 or not np.isfinite([exponent, cutoff]).all() or exponent < 0 or not 0 < cutoff <= 1:
        raise ValueError('Invalid fractal wavetable')
    frequency = np.arange(1, count//2 + 1)
    spectrum = np.zeros(count//2 + 1, dtype=complex)
    phase = np.random.default_rng(seed).uniform(0, 2*np.pi, len(frequency))
    spectrum[1:] = frequency**(-exponent/2) * np.exp(1j*phase)
    spectrum[1:][frequency > cutoff * count/2] = 0
    spectrum[-1] = 0
    result = np.fft.irfft(spectrum, n=count)
    rms = np.sqrt(np.mean(result**2))
    if rms == 0:
        raise ValueError('Wavetable cutoff excludes all frequencies')
    return result / rms
