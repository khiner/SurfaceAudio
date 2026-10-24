#!/usr/bin/env python3
"""Independent punch data, elastic-layer limits, and the complete surface/contact pipeline."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.optimize import root

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from andersson_model import causal_response, layer_compliance, layer_green, layer_kernel, punch_factor, surface_springs

BINARY = ROOT / 'build/anderssonReproduce'


class AnderssonTest(unittest.TestCase):
    def test_published_punch(self):
        data = np.genfromtxt(ROOT / 'tests/data/andersson_punch.csv', delimiter=',', comments='#', skip_header=4, names=True)
        errors = []
        for row in data:
            value = punch_factor(row['ratio'], row['poisson'], bool(row['bonded']))
            errors.append(abs(value / row['normalized_stiffness'] - 1))
        # The independent author BEM data have finite spatial resolution; this is not the quadrature tolerance.
        self.assertLess(max(errors), .003)
        print(f'Published flat-punch BEM: maximum relative error {max(errors):.6g}', flush=True)
        self.assertEqual(punch_factor(0), 1)
        self.assertLess(abs(punch_factor(1e-7) - 1), 2e-7)

    def test_static_layer(self):
        k = np.geomspace(.001, 1e5, 64)
        for nu in (.0, .3, .4999):
            actual = layer_compliance(k, 0, 25e6, nu, 1300, .01)
            expected = 2 * (1 - nu**2) / (25e6 * k) * layer_kernel(k * .01, nu)
            np.testing.assert_allclose(actual, expected, rtol=2e-8)
            zero = layer_compliance(0, 0, 25e6, nu, 1300, .01)
            self.assertAlmostEqual(float(zero.real / (.01 * (1 + nu) * (1 - 2 * nu) / (25e6 * (1 - nu)))), 1, places=11)

    def test_uniform_dynamic_layer(self):
        young, nu, rho, thickness = 35e6 + 8e6j, .49, 1300, .01
        longitudinal = young * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
        speed = np.sqrt(longitudinal / rho)
        for omega in (1, 3000, 50000, 150000):
            expected = np.tan(omega * thickness / speed) / (rho * speed * omega)
            actual = layer_compliance(0, omega, young, nu, rho, thickness)
            self.assertLess(abs(actual / expected - 1), 2e-10)
            self.assertLess(actual.imag, 0)

    def test_dynamic_layer_boundary_problem(self):
        # Independent dimensional Navier transfer and simultaneous top/bottom boundary solve, without slab condensation.
        young, nu, rho, thickness = 25e6 + 3e6j, .45, 1300, .01
        mu, lam = young / (2 * (1 + nu)), young * nu / ((1 + nu) * (1 - 2 * nu))
        for k in (1., 50., 400.):
            for omega in (100., 5000., 100000.):
                matrix = np.array([[0, -1j * k, 1 / mu, 0], [-1j * k * lam / (lam + 2 * mu), 0, 0, 1 / (lam + 2 * mu)],
                                   [4 * mu * (lam + mu) / (lam + 2 * mu) * k*k - rho * omega**2, 0, 0, -1j * k * lam / (lam + 2 * mu)],
                                   [0, -rho * omega**2, -1j * k, 0]])
                transfer = expm(thickness * matrix)
                displacement = np.linalg.solve(transfer[:2, :2], transfer[:2, 3])
                actual = layer_compliance(k, omega, young, nu, rho, thickness)
                self.assertLess(abs(actual / displacement[1] - 1), 2e-9)

    def test_causal_transform(self):
        # A damped single-degree-of-freedom compliance provides an independent cosine-transform oracle.
        taps, rate = 256, 51200
        omega = 2 * np.pi * np.linspace(0, rate / 2, taps)
        response = 1 / (3e6 - .1 * omega**2 + 1j * 500 * omega)
        green = causal_response(response, rate)
        self.assertLess(abs(green.sum() / rate / response[0] - 1), 1e-14)
        for i in (0, 1, 17, taps - 1):
            weight = np.ones(taps); weight[[0, -1]] = .5
            expected = np.sum(weight * response.real * np.cos(np.pi * np.arange(taps) * i / (taps - 1))) / (taps - 1)
            expected *= rate * (1 if i in (0, taps - 1) else 2)
            self.assertAlmostEqual(green[i], expected, places=16)

    def test_geometry(self):
        # The 1 mm element cuts the 0.6 mm scan pixels: areas are .36,.24,.24,.16 mm^2.
        heights = np.array([[.0004, .0002], [.0001, 0.]])
        tops, tables = surface_springs(heights, .0006, (1, 1), .001, 25e6, True, .75, 7)
        self.assertEqual(tops[0], .0004)
        np.testing.assert_array_equal(tables[0, 0], [0, 0])
        for compression, stiffness in tables[0, [1, 2, 3, 5]]:
            area = np.array([[.36, .24], [.24, .16]])[heights > .0004 - compression].sum() * 1e-6
            radius = np.sqrt(area / np.pi)
            expected = 2 * 25e6 / .75 * radius * punch_factor(radius / (.0004 - compression))
            self.assertLess(abs(stiffness / expected - 1), 4e-5)
        shifted, shifted_table = surface_springs(heights + .1, .0006, (1, 1), .001, 25e6, True, .75, 7)
        np.testing.assert_allclose(shifted - tops, .1)
        # Avoid exact height thresholds in this invariance check; their Heaviside value is convention-dependent.
        np.testing.assert_allclose(shifted_table[:, :, 0], tables[:, :, 0], rtol=1e-12)
        np.testing.assert_allclose(shifted_table[:, [1, 3, 5], 1], tables[:, [1, 3, 5], 1], rtol=4e-5)
        with self.assertRaises(ValueError):
            surface_springs(np.zeros((4, 4)), .001, (1, 1), .001, 25e6, True)
        # The 19 mm boundary coincides with scan pixel 500; a zero-area neighbour must not set the element height.
        scan = np.tile(np.arange(527) * .0001, (27, 1))
        scan[:, 500] = .1
        tops, _ = surface_springs(scan, 38e-6, (1, 20), .001, 25e6, True, knots=3)
        self.assertEqual(tops[18], scan[0, 499])
        self.assertEqual(tops[19], scan[0, 500])

    def test_layer_spatial_convergence(self):
        material = np.array([[0, 25e6, 0], [25600, 50e6, 25e6]])
        # Test numerical domain and source-element resolution separately, without changing the material.
        _, low = layer_green((2, 2), .001, .01, .49, 1300, material, 51200, 3, 32, 8)
        _, fine = layer_green((2, 2), .001, .01, .49, 1300, material, 51200, 3, 32, 16)
        _, wide = layer_green((2, 2), .001, .01, .49, 1300, material, 51200, 3, 64, 16)
        resolution = np.max(abs(low - fine)) / np.max(abs(fine))
        domain = np.max(abs(fine[0] - wide[0])) / np.max(abs(wide[0]))
        print(f'Layer refinement: spatial cutoff {resolution:.6g}, static domain {domain:.6g}', flush=True)
        self.assertLess(resolution, .015)
        self.assertLess(domain, .005)

    def test_complete_pipeline(self):
        heights = np.array([[.0001, .00005, .00009, .00002], [0, .00008, .00001, .00007],
                            [.00006, .00003, .00008, .00004], [.00002, 0, .00005, .00001]])
        tops, tables = surface_springs(heights, .0005, (2, 2), .001, 25e6, True, .9, 513)
        material = np.array([[0, 25e6, 0], [3000, 35e6, 15e6], [25600, 50e6, 25e6]])
        green, response = layer_green((2, 2), .001, .01, .49, 1300, material, 51200, 64, 16, 8)
        y, x = np.indices((2, 2)).reshape(2, -1)
        matrix = response[0].real[np.abs(y[:, None] - y), np.abs(x[:, None] - x)]
        indentation = 3e-5
        approach = indentation + tops - tops.max()
        def forces(compression):
            return np.array([quad(lambda z: np.interp(z, table[:, 0], table[:, 1]), 0, max(0, c),
                                  points=table[(table[:, 0] > 0) & (table[:, 0] < max(0, c)), 0], epsabs=1e-12, limit=1024)[0]
                             for c, table in zip(compression, tables)])
        expected = root(lambda c: c + matrix @ forces(c) - approach, np.zeros(4), tol=1e-10)
        self.assertTrue(expected.success)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'dimensions.txt').write_text('2 2 64 513 51200\n')
            for name, array in [('heights', tops), ('springs', tables), ('green', green), ('schedule', np.array([[0, 256], [indentation, 4096]]))]:
                np.asarray(array, dtype='<f8').tofile(directory / f'{name}.f64')
            subprocess.run([str(BINARY), str(directory), str(directory / 'static'), '--equilibrium'], check=True)
            actual = np.fromfile(directory / 'static/final_force.f64')
            np.testing.assert_allclose(actual, forces(expected.x), rtol=1e-7, atol=1e-9)
            subprocess.run([str(BINARY), str(directory), str(directory / 'dynamic')], check=True)
            dynamic = np.fromfile(directory / 'dynamic/final_force.f64')
            np.testing.assert_allclose(dynamic, actual, rtol=1e-6, atol=1e-9)
            print(f'End-to-end settled force: {actual.sum():.9g} N, transient difference {np.max(abs(dynamic-actual)):.6g} N', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, default=BINARY)
    args, remaining = parser.parse_known_args()
    BINARY = args.binary.resolve()
    unittest.main(argv=[sys.argv[0], *remaining])
