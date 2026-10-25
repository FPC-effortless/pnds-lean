"""Exact-law and adverse-case diagnostics; no learned model or real-data claim."""
import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "foundation/v1"))
from native_memory_math import (write, read, frobenius, query_decomposition,
    self_read_residual, finite_difference_value_sensitivity)


def unit(values):
    norm = math.hypot(*values)
    return [v / norm for v in values]


class NativeDeltaMemory(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(14124)
        self.empty = [[0.0] * 8 for _ in range(8)]
        self.k = [1.0] + [0.0] * 7
        self.q = [0.0, 1.0] + [0.0] * 6

    def test_read_only_preserves_original_byte_values_even_with_decay(self):
        m = [[self.rng.uniform(-1, 1) for _ in range(8)] for _ in range(8)]
        result = write(m, self.k, [0.1]*8, 0.001, 1.0, writable=False)
        self.assertEqual(result, m)

    def test_full_write_self_key_and_residual_unprojected(self):
        for _ in range(120):
            m = [[self.rng.uniform(-.2, .2) for _ in range(8)] for _ in range(8)]
            k = unit([self.rng.uniform(-1, 1) for _ in range(8)])
            v = [self.rng.uniform(-1, 1) for _ in range(8)]
            a, b = self.rng.random(), self.rng.random()
            new = read(write(m, k, v, a, b, radius=None), k)
            e = self_read_residual(m, k, v, a, b)
            for target, actual, expected in zip(v, new, e):
                self.assertAlmostEqual(target-actual, expected, places=11)

    def test_arbitrary_query_interference_identity(self):
        for _ in range(120):
            m = [[self.rng.uniform(-.2, .2) for _ in range(8)] for _ in range(8)]
            k = unit([self.rng.uniform(-1, 1) for _ in range(8)])
            q = unit([self.rng.uniform(-1, 1) for _ in range(8)])
            v = [self.rng.uniform(-1, 1) for _ in range(8)]
            a, b = self.rng.random(), self.rng.random()
            actual = read(write(m, k, v, a, b, radius=None), q)
            expected = query_decomposition(m, k, v, a, b, q)
            for x, y in zip(actual, expected):
                self.assertAlmostEqual(x, y, places=11)

    def test_orthogonal_write_still_decays_old_read(self):
        m = [[0.0]*8 for _ in range(8)]
        m[0][1] = 1.0
        for _ in range(256):
            m = write(m, self.k, [0.0]*8, .99, .2, radius=None)
        self.assertAlmostEqual(read(m, self.q)[0], .99**256, places=11)
        self.assertLess(read(m, self.q)[0], .1)

    def test_projection_counters_exact_full_strength_correction(self):
        v = [5.0]+[0.0]*7
        exact = write(self.empty, self.k, v, 1.0, 1.0, radius=None)
        clipped = write(self.empty, self.k, v, 1.0, 1.0, radius=4.0)
        self.assertAlmostEqual(read(exact, self.k)[0], 5.0)
        self.assertAlmostEqual(read(clipped, self.k)[0], 4.0)
        self.assertLessEqual(frobenius(clipped), 4.0 + 1e-12)

    def test_finite_difference_and_analytic_value_derivative(self):
        for _ in range(70):
            m = [[self.rng.uniform(-.2,.2) for _ in range(8)] for _ in range(8)]
            k = unit([self.rng.uniform(-1,1) for _ in range(8)])
            q = unit([self.rng.uniform(-1,1) for _ in range(8)])
            v = [self.rng.uniform(-1,1) for _ in range(8)]
            empirical, analytic = finite_difference_value_sensitivity(
                m,k,v,self.rng.random(),self.rng.random(),q,3)
            self.assertAlmostEqual(empirical, analytic, places=8)

    def test_frobenius_radius_and_unsafe_inputs(self):
        m = [[self.rng.uniform(-20,20) for _ in range(8)] for _ in range(8)]
        new = write(m,self.k,[.1]*8,.9,.5,radius=4.0)
        self.assertLessEqual(frobenius(new),4.0+1e-11)
        with self.assertRaises(ValueError):
            write(m,self.k,[.1]*8,1.1,.5)
        with self.assertRaises(ValueError):
            write(m,self.k,[float("nan")]*8,.9,.5)
        with self.assertRaises(ValueError):
            write(m,self.k,[.1]*8,.9,.5,radius=-1)


if __name__ == "__main__":
    unittest.main()
