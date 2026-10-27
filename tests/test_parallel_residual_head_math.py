"""Constructed scalar property checks; no real-world claim."""
import math
import random
import unittest

from foundation.v1.parallel_residual_head_math import predict, stable_sigmoid


class PostScanScalarTests(unittest.TestCase):
    def test_boundedness_for_declared_input_domain(self):
        rng = random.Random(271027)
        for _ in range(500):
            base = rng.uniform(-200.0, 200.0)
            scale = rng.uniform(0.0, 3.0)
            logit = rng.uniform(-25.0, 25.0)
            innovation = rng.uniform(-100.0, 100.0)
            y = predict(base, scale, logit, innovation)
            self.assertLessEqual(abs(y-base), scale + 1e-12)

    def test_zero_innovation_and_stable_extreme_logits(self):
        for logit in (-1000.0, 0.0, 1000.0):
            self.assertEqual(predict(4.0, 2.0, logit, 0.0), 4.0)
            self.assertTrue(0.0 <= stable_sigmoid(logit) <= 1.0)
        self.assertEqual(stable_sigmoid(-1000.0), 0.0)

    def test_reject_nonfinite_or_negative_scale(self):
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                predict(0.0, 1.0, 0.0, bad)
        with self.assertRaises(ValueError):
            predict(0.0, -0.1, 0.0, 1.0)
        with self.assertRaises(ValueError):
            predict(0.0, 1.0, math.inf, 1.0)


if __name__ == "__main__":
    unittest.main()
