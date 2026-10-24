"""Andersson-Kropp (2008) surface springs and rigid-backed elastic-layer response, in SI units."""
from functools import lru_cache

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.linalg import expm
from scipy.special import roots_legendre


def layer_kernel(t, poisson, bonded=True):
    """Static normal compliance factor; Forsbach (2023), Equations 29 and 31."""
    t = np.asarray(t, dtype=float)
    e = np.exp(-2 * t)
    if not bonded:
        return np.divide((-np.expm1(-2 * t))**2, 1 - e * e + 4 * t * e, out=np.zeros_like(t), where=t != 0)
    kappa = 3 - 4 * poisson
    # sinh(2t)-2t loses precision for a nearly incompressible layer at small t.
    small = t < .01
    remainder = np.zeros_like(t)
    remainder[small] = (2 * t[small])**3 / 6 * (1 + (2 * t[small])**2 / 20 + (2 * t[small])**4 / 840)
    numerator = kappa * (1 - e * e) - 4 * t * e
    numerator = np.where(small, 2 * e * (kappa * remainder + 2 * (kappa - 1) * t), numerator)
    return numerator / (kappa * (1 + e * e) + (1 + 4 * t * t + kappa * kappa) * e)


@lru_cache(maxsize=32)
def gauss(order):
    x, w = roots_legendre(order)
    return (x + 1) / 2, w / 2


def punch_factor(ratio, poisson=.5, bonded=True, tolerance=1e-8):
    """K/(2 E* a) from the axisymmetric flat-punch Fredholm equation, E*=E/(1-nu^2)."""
    if not np.isfinite(ratio) or ratio < 0 or not 0 <= poisson <= .5 or not 0 < tolerance < 1:
        raise ValueError("Invalid punch confinement, Poisson ratio or tolerance")
    if ratio == 0:
        return 1.
    # The correction kernel decays exponentially beyond kh=24; no thin-film stiffness cap is applied.
    order = max(24, int(np.ceil(4 * ratio)))
    previous = None
    while order <= 1536:
        x, w = gauss(order)
        t, v = gauss(max(128, int(np.ceil(32 * ratio))))
        t, v = 24 * t, 24 * v
        basis = np.cos(np.outer(x, ratio * t))
        kernel = (2 * ratio / np.pi) * np.einsum('it,t,jt->ij', basis, (layer_kernel(t, poisson, bonded) - 1) * v, basis, optimize=True)
        u = np.linalg.solve(np.eye(order) + kernel * w, np.ones(order))
        value = float(np.sum(w * u))
        if not np.isfinite(value) or value <= 0:
            raise ArithmeticError("Flat-punch quadrature lost positive stiffness")
        if previous is not None and abs(value / previous - 1) <= tolerance:
            return value
        previous, order = value, 2 * order
    raise ArithmeticError(f"Flat-punch quadrature did not converge for a/h={ratio:g}; increase quadrature resolution")


def punch_curve(maximum_ratio, bonded, tolerance=2e-5):
    """Adaptively tabulate the incompressible punch solution in log confinement."""
    if not np.isfinite(maximum_ratio) or maximum_ratio <= 0:
        raise ValueError("Positive finite maximum confinement required")
    low, high = np.log(1e-6), np.log(max(maximum_ratio, 1e-5))
    x = np.linspace(low, high, 17)
    y = np.log([punch_factor(np.exp(q), bonded=bonded) for q in x])
    for _ in range(12):
        curve = PchipInterpolator(x, y)
        mid = (x[:-1] + x[1:]) / 2
        truth = np.log([punch_factor(np.exp(q), bonded=bonded) for q in mid])
        failed = np.abs(curve(mid) - truth) > tolerance
        if not np.any(failed):
            def evaluate(ratio):
                ratio = np.asarray(ratio)
                if np.any(ratio < 0) or np.any(ratio > maximum_ratio * (1 + 1e-12)):
                    raise ValueError("Punch interpolation outside prepared confinement range")
                result = np.exp(curve(np.log(np.maximum(ratio, np.exp(low)))))
                # This branch evaluates the same Fredholm solution, including the half-space limit.
                small = ratio < np.exp(low)
                if np.any(small):
                    result[small] = [punch_factor(float(r), bonded=bonded) for r in ratio[small]]
                return result
            return evaluate
        x, y = np.r_[x, mid[failed]], np.r_[y, truth[failed]]
        order = np.argsort(x)
        x, y = x[order], y[order]
    raise ArithmeticError("Flat-punch interpolation did not converge")


def surface_springs(heights, scan_spacing, elements, element_spacing, young, bonded, fraction=.9, knots=257):
    """Equations 34-36, including fractional scan-pixel areas at coarse element boundaries."""
    heights = np.asarray(heights, dtype=float)
    ny, nx = elements
    if (heights.ndim != 2 or not np.all(np.isfinite(heights)) or not np.isfinite(scan_spacing) or scan_spacing <= 0
            or not np.isfinite(element_spacing) or element_spacing <= 0 or min(nx, ny) < 1
            or not np.isfinite(young) or young <= 0 or not 0 < fraction < 1 or knots < 3):
        raise ValueError("Invalid surface, geometry, modulus or spring resolution")
    if heights.shape[0] * scan_spacing < ny * element_spacing or heights.shape[1] * scan_spacing < nx * element_spacing:
        raise ValueError("Measured surface does not cover the contact patch")
    axes = []
    for count, fine_count in zip((ny, nx), heights.shape):
        coarse = np.arange(count + 1)[:, None] * element_spacing
        fine = np.arange(fine_count + 1)[None, :] * scan_spacing
        axes.append(np.maximum(0, np.minimum(coarse[1:], fine[:, 1:]) - np.maximum(coarse[:-1], fine[:, :-1])))
    records = []
    for iy in range(ny):
        for ix in range(nx):
            weight = np.outer(axes[0][iy], axes[1][ix])
            mask = weight > 0
            z, area = heights[mask], weight[mask]
            ordering = np.argsort(-z)
            z, area = z[ordering], area[ordering]
            depth = z[0] - z[-1]
            if depth <= 0:
                raise ValueError(f"Element ({ix},{iy}) is exactly flat; its interface is a rigid constraint, not a finite roughness spring")
            compression = np.linspace(0, fraction * depth, knots)
            count = np.searchsorted(z[0] - z, compression, side='left')
            contact_area = np.r_[0., np.cumsum(area)][count]
            radius = np.sqrt(contact_area / np.pi)
            thickness = depth - compression
            ratio = radius / thickness
            records.append((z[0], compression, radius, ratio))
    tops, compression, radius, ratio = map(np.asarray, zip(*records))
    stiffness = 2 * young / .75 * radius * punch_curve(float(ratio.max()), bonded)(ratio)
    return tops, np.stack((compression, stiffness), axis=-1)


def layer_compliance(wavenumber, omega, young, poisson, density, thickness):
    """Normal displacement / pressure for a shear-free surface and bonded rigid base.

    Condense the Navier equations in depth using elastic dynamic stiffness matrices.
    Doubling thin slabs avoids both growing evanescent waves and degenerate P/S potentials at DC.
    Time convention is exp(+i omega t), so passive material has Im(E)>=0.
    """
    k = np.asarray(wavenumber, dtype=float)
    if (not np.all(np.isfinite(k)) or np.any(k < 0) or not np.isfinite(omega) or omega < 0
            or not np.isfinite(young) or np.real(young) <= 0 or np.imag(young) < 0
            or not 0 <= poisson < .5 or not np.isfinite(density) or density <= 0
            or not np.isfinite(thickness) or thickness <= 0):
        raise ValueError("Invalid elastic-layer parameters")
    q = k.ravel() * thickness
    mu = young / (2 * (1 + poisson))
    longitudinal = 2 * (1 - poisson) / (1 - 2 * poisson)
    gamma = 1 - 2 / longitudinal
    frequency = density * omega**2 * thickness**2 / mu
    scale = np.maximum(np.maximum(1, q), abs(frequency)**.5)
    matrix = np.zeros((*q.shape, 4, 4), dtype=complex)
    matrix[..., 0, 1], matrix[..., 0, 2] = -1j * q, scale
    matrix[..., 1, 0], matrix[..., 1, 3] = -1j * gamma * q, scale / longitudinal
    matrix[..., 2, 0], matrix[..., 2, 3] = (4 * (longitudinal - 1) / longitudinal * q**2 - frequency) / scale, -1j * gamma * q
    matrix[..., 3, 1], matrix[..., 3, 2] = -frequency / scale, -1j * q
    levels = np.ceil(np.log2(2 * scale)).astype(int)
    transfer = expm(matrix / (2.**levels)[:, None, None])
    a = np.linalg.solve(transfer[..., :2, 2:], transfer[..., :2, :2])
    b = -np.linalg.inv(transfer[..., :2, 2:])
    c = transfer[..., 2:, :2] - transfer[..., 2:, 2:] @ a
    d = -transfer[..., 2:, 2:] @ b
    for level in range(int(levels.max())):
        active = levels > level
        aa, bb, cc, dd = a[active], b[active], c[active], d[active]
        u, v = np.linalg.solve(dd + aa, cc), np.linalg.solve(dd + aa, bb)
        a[active], b[active], c[active], d[active] = aa - bb @ u, -bb @ v, -cc @ u, dd - cc @ v
    result = (np.linalg.inv(a)[..., 1, 1] * thickness / (mu * scale)).reshape(k.shape)
    if not np.all(np.isfinite(result)):
        raise ArithmeticError("Nonfinite elastic-layer response")
    return result


def causal_response(frequency_response, sample_rate):
    """Two-times-real construction, retaining half weight at DC-time and Nyquist-time endpoints."""
    response = np.asarray(frequency_response)
    if response.shape[0] < 2 or not np.all(np.isfinite(response)) or not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Invalid frequency response or sample rate")
    even = np.fft.irfft(response.real, n=2 * (len(response) - 1), axis=0)
    green = even[:len(response)].copy()
    green[1:-1] *= 2
    return green * sample_rate


def layer_green(elements, spacing, thickness, poisson, density, modulus, sample_rate, taps, domain_elements, oversampling):
    """Uniform square force elements, center receivers, periodic lateral calculation domain.

    modulus has columns frequency [Hz], storage modulus [Pa], loss modulus [Pa].
    Returns compact [lag, abs(dy), abs(dx)] Green functions in m/(N s).
    Domain extent and spatial cutoff are explicit convergence parameters, not paper input data.
    """
    ny, nx = elements
    material = np.asarray(modulus, dtype=float)
    if (taps < 2 or min(nx, ny) < 1 or domain_elements < 2 * max(elements) or oversampling < 1
            or not np.isfinite(spacing) or spacing <= 0 or not np.isfinite(sample_rate) or sample_rate <= 0
            or material.ndim != 2 or material.shape[1] != 3 or len(material) < 2 or not np.all(np.isfinite(material))
            or material[0, 0] != 0 or material[-1, 0] < sample_rate / 2 or np.any(np.diff(material[:, 0]) <= 0)
            or np.any(material[:, 1] <= 0) or np.any(material[:, 2] < 0) or material[0, 2] != 0):
        raise ValueError("Invalid layer grid or material table; DC through Nyquist must be supplied")
    size = domain_elements * oversampling
    fine_spacing = spacing / oversampling
    k = 2 * np.pi * np.fft.fftfreq(size, fine_spacing)
    radius, inverse = np.unique(np.hypot(k[:, None], k[None, :]), return_inverse=True)
    source = np.sinc(k * spacing / (2 * np.pi))
    source = source[:, None] * source[None, :]
    frequencies = np.linspace(0, sample_rate / 2, taps)
    young = np.interp(frequencies, material[:, 0], material[:, 1]) + 1j * np.interp(frequencies, material[:, 0], material[:, 2])
    response = np.empty((taps, ny, nx), complex)
    for i, (frequency, elastic) in enumerate(zip(frequencies, young)):
        radial = layer_compliance(radius, 2 * np.pi * frequency, elastic, poisson, density, thickness)
        spectrum = radial[inverse.reshape((size, size))] * source
        spatial = np.fft.ifft2(spectrum) / fine_spacing**2
        response[i] = spatial[:ny * oversampling:oversampling, :nx * oversampling:oversampling]
    return causal_response(response, sample_rate), response
