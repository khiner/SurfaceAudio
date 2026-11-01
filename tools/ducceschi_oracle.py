"""Independent mechanical-coordinate solve for the coupled instruments."""
import numpy as np
from scipy import sparse
from scipy.optimize import newton
from scipy.sparse.linalg import spsolve


def mechanical_oracle(model, frames):
    c = len(model.barrier)
    dt = 1/model.rate
    q, previous = model.q.copy(), model.previous.copy()
    psi = model.psi.copy()
    j = model.contact
    diagonal = model.mass+dt*model.damping/2
    bilateral = model.bilateral.astype(bool)
    output, first_force = [], []
    peak_contact_energy = 0.
    previous_bow_velocity = -model.bow_speed
    for _ in range(frames):
        rhs = 2*model.mass*q-(model.mass-dt*model.damping/2)*previous-dt**2*(model.stiffness@q)-dt*(model.loss@(q-previous))
        if model.bow_force:
            response = np.sum(model.bow**2/diagonal)*dt/2
            free = np.sum(model.bow*((rhs/diagonal-previous)/(2*dt)))-model.bow_speed
            friction = lambda v: np.sqrt(200)*v*np.exp(.5-100*v*v)
            residual = lambda v: v+response*model.bow_force*friction(v)-free
            derivative = lambda v: 1+response*model.bow_force*np.sqrt(200)*np.exp(.5-100*v*v)*(1-200*v*v)
            velocity = newton(residual, previous_bow_velocity, fprime=derivative, tol=1e-12, maxiter=100)
            previous_bow_velocity = velocity
            force = -model.bow_force*friction(velocity)
            rhs += dt**2*model.bow*force
        eta = model.barrier+j@q
        free_change = j@(rhs/diagonal-previous)
        g = np.zeros(c)
        touching = (eta > 0) | ((model.rule == 2) & (eta == 0)) | (bilateral & (eta != 0))
        g[touching] = np.sqrt(.5*model.kappa[touching]*(model.exponent[touching]+1))*np.abs(eta[touching])**((model.exponent[touching]-1)/2)
        g[(psi < 0) & ~bilateral] *= -1
        g[bilateral & (eta < 0)] *= -1
        separated = ~touching & ~bilateral & (free_change != 0) & ((eta < 0) | (model.rule == 1))
        g[separated] = -2*psi[separated]/free_change[separated]
        g[touching & (model.rule == 1) & (g*(model.barrier+j@previous)/4 > psi)] = 0
        matrix = sparse.diags(diagonal)+dt**2/4*(j.T@sparse.diags(g*g)@j)
        rhs += dt**2*(j.T@(g*g*(j@previous)/4-g*psi))
        following = spsolve(matrix.tocsc(), rhs)
        following_psi = psi+g*(j@(following-previous))/2
        peak_contact_energy = max(peak_contact_energy, float(np.sum(following_psi**2)/2))
        first_force.append(g[0]*(psi[0]+following_psi[0])/2)
        pickup = q if model.smoothing == 0 else ((following+q)/2 if model.smoothing == 1 else (following+2*q+previous)/4)
        output.append(np.sum(model.pickup*pickup))
        previous, q, psi = q, following, following_psi
    return np.array(output), np.array(first_force), q, psi, peak_contact_energy
