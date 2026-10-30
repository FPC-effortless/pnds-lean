"""Constructed operator checks only; no capability or source-parity claim."""
import unittest

import numpy as np

from langmath095_reference import (
    bounded_aligned_step, bounded_log_position, compose, concept_solve,
    fixed_key_gain, serial,
)


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(9501)

    def pair(self):
        return self.rng.normal(size=(3, 3)), self.rng.normal(size=(2, 3))

    def test_noncommuting_chronological_action_and_associativity(self):
        a, b, c = self.pair(), self.pair(), self.pair()
        s = self.rng.normal(size=(2, 3))
        p, q = compose(a, b)
        np.testing.assert_allclose(s @ p + q, (s @ a[0] + a[1]) @ b[0] + b[1])
        left, right = compose(compose(a, b), c), compose(a, compose(b, c))
        for x, y in zip(left, right):
            np.testing.assert_allclose(x, y, rtol=1e-13, atol=1e-13)
        self.assertGreater(np.max(np.abs(a[0] @ b[0] - b[0] @ a[0])), .01)

    def test_identity_and_empty_sequence(self):
        pair = self.pair()
        identity = np.eye(3), np.zeros((2, 3))
        for actual in (compose(identity, pair), compose(pair, identity)):
            for x, y in zip(actual, pair):
                np.testing.assert_array_equal(x, y)
        self.assertEqual(serial(np.empty((0, 3, 3)), np.empty((0, 2, 3)),
                                np.ones((2, 3))).shape, (0, 2, 3))

    def test_serial_chunk_law_with_nonzero_initial(self):
        p = self.rng.normal(size=(9, 3, 3)) * .1
        q = self.rng.normal(size=(9, 2, 3))
        initial = self.rng.normal(size=(2, 3))
        expected = serial(p, q, initial)
        for cut in range(1, len(p)):
            head = serial(p[:cut], q[:cut], initial)
            tail = serial(p[cut:], q[cut:], head[-1])
            np.testing.assert_array_equal(np.concatenate((head, tail)), expected)

    def test_bounded_write_with_rotating_keys_and_masks(self):
        m = np.zeros((16, 8))
        for _ in range(2000):
            key = self.rng.normal(size=8)
            key /= np.linalg.norm(key)
            value = np.tanh(self.rng.normal(size=16))
            gates = self.rng.uniform(size=3)
            before = m.copy()
            m = bounded_aligned_step(m, key, value, *gates)
            self.assertLessEqual(np.linalg.norm(m), 4 + 1e-12)
            np.testing.assert_array_equal(bounded_aligned_step(before, key, value,
                                          *gates, admit=False), before)

    def test_bound_rejects_invalid_assumptions(self):
        with self.assertRaises(ValueError):
            bounded_aligned_step(np.zeros((2, 2)), np.ones(2), np.ones(2), .9, .5, .5)
        with self.assertRaises(ValueError):
            fixed_key_gain(1., 0., .5)

    def test_original_gate_amplification_witness(self):
        gain = fixed_key_gain(.9999, .0001, .5)
        self.assertGreater(gain, 2500)
        self.assertLess(gain, 2501)
        alpha, erase, write, value = .9999, .0001, .5, .2
        equilibrium = gain * value
        self.assertAlmostEqual(alpha * (1 - erase) * equilibrium + write * value,
                               equilibrium, places=11)

    def test_concept_conditioning_and_uniform_pool_cancellation(self):
        for _ in range(20):
            g = np.tanh(self.rng.normal(size=(6, 16)))
            prior = np.tanh(self.rng.normal(size=(6, 16)))
            c, laplacian, h = concept_solve(g, prior, self.rng.normal(size=(6, 6)))
            eig = np.linalg.eigvalsh(h)
            self.assertGreaterEqual(eig.min(), 1.25 - 1e-12)
            self.assertLessEqual(eig.max() / eig.min(), 2.2 + 1e-12)
            np.testing.assert_allclose(laplacian @ np.ones(6), 0, atol=1e-14)
            np.testing.assert_allclose(c.mean(axis=0), .8 * (g + .25 * prior).mean(axis=0),
                                       atol=1e-13, rtol=1e-13)
            self.assertLessEqual(np.abs(c).max(), 1 + 1e-12)

    def test_bounded_position_range_and_invalid_inputs(self):
        values = bounded_log_position(np.array([0, 1, 1024, 2**63 - 1]))
        self.assertTrue(np.all(values >= 0))
        self.assertTrue(np.all(values < 1))
        self.assertTrue(np.all(np.diff(values) > 0))
        for invalid in (-1, float("nan"), float("inf"), 1j, "3"):
            with self.assertRaises(ValueError):
                bounded_log_position(invalid)

    def test_floating_point_regrouping_is_not_bitwise_associative(self):
        p = np.full(3, .5, dtype=np.float32)
        q = np.array([.5, -.25, 1e-8], dtype=np.float32)
        serial_value = np.float32(0)
        for pi, qi in zip(p, q):
            serial_value = serial_value * pi + qi
        grouped = (q[0] * (p[1] * p[2])) + (q[1] * p[2] + q[2])
        self.assertNotEqual(serial_value, grouped)


if __name__ == "__main__":
    unittest.main()
