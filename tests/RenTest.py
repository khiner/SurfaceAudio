#!/usr/bin/env python3
"""Check the spring model against rigid-motion invariance and direct matrix dynamics."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from ren_model import fractal_table, mesh_modes, normal_map, spring_system


class RenTest(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[0., 0, 0], [.1, 0, 0], [0, .1, 0], [0, 0, .1]])
        self.triangles = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
        self.material = (1e6, 1000, .002)

    def test_springs(self):
        mass, k = spring_system(self.vertices, self.triangles, *self.material)
        areas = [.005, .005, .005, .005*np.sqrt(3)]
        self.assertAlmostEqual(mass.sum()/3, sum(areas)*1000*.002)
        self.assertGreater(np.ptp(mass), 0)
        np.testing.assert_allclose(k, k.T, atol=1e-12)
        for axis in np.eye(3):
            for motion in (np.tile(axis, 4), np.cross(np.tile(axis, (4, 1)), self.vertices).ravel()):
                np.testing.assert_allclose(k @ motion, 0, atol=1e-10)
        self.assertEqual(np.linalg.matrix_rank(k), 6)

    def test_modal_impulse_against_coupled_dynamics(self):
        mass, k = spring_system(self.vertices, self.triangles, *self.material)
        gamma, eta = 12., 2e-6
        pickup = np.zeros(12)
        pickup[2] = 1
        modes, shapes, checks = mesh_modes(self.vertices, self.triangles, *self.material, gamma, eta, pickup)
        self.assertEqual(len(modes), 6)
        self.assertLess(checks['mass_orthogonality_error'], 1e-12)
        # Opposite impulses along an edge have no rigid-body translation or rotation component.
        impulse = np.zeros(12)
        impulse[0], impulse[3] = -1, 1
        dynamics = np.block([[np.zeros_like(k), np.eye(12)], [-k/mass[:, None], -gamma*np.eye(12)-eta*k/mass[:, None]]])
        state = np.r_[np.zeros(12), impulse/mass]
        times = np.array([.0001, .001, .003, .01])
        integrated = solve_ivp(lambda t, state: dynamics @ state, (0, times[-1]), state, t_eval=times, method='DOP853', rtol=1e-11, atol=1e-13)
        self.assertTrue(integrated.success)
        for t, value in zip(times, integrated.y.T):
            direct = pickup @ value[:12]
            modal = np.sum(modes[:, 2] * (shapes.T @ impulse) * np.exp(-t/modes[:, 1]) * np.sin(2*np.pi*modes[:, 0]*t))
            self.assertAlmostEqual(direct, modal, delta=2e-11)

    def test_spatial_spectrum_and_normals(self):
        table = fractal_table(4096, 1.4, 2010)
        self.assertAlmostEqual(np.mean(table), 0)
        self.assertAlmostEqual(np.mean(table**2), 1)
        np.testing.assert_array_equal(table, fractal_table(4096, 1.4, 2010))
        slope = np.polyfit(np.log(np.arange(1, 1000)), np.log(np.abs(np.fft.rfft(table)[1:1000])**2), 1)[0]
        self.assertAlmostEqual(slope, -1.4, places=10)
        n = 128
        x = np.arange(n)/n
        height = np.tile(.001*np.sin(2*np.pi*x), (n, 1))
        normals = normal_map(height, [.1, .1])
        expected = .001*np.sin(2*np.pi/n)*n/.1*np.cos(2*np.pi*x)
        np.testing.assert_allclose(-normals[0, :, 0]/normals[0, :, 2], expected, atol=1e-14)
        np.testing.assert_array_equal(normals[:, :, 1], 0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            spring_system(self.vertices, [[0, 0, 1]], *self.material)
        with self.assertRaises(ValueError):
            normal_map(np.zeros((4, 4)), [0, 1])
        with self.assertRaises(ValueError):
            fractal_table(4096, 1, 0, 1e-9)


if __name__ == '__main__':
    unittest.main()
