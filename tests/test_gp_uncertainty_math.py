"""No-fit PNDS scalar GP/precision properties, not trained model claims."""
import importlib.util
import math
from pathlib import Path
import random
import unittest

MODULE = Path(__file__).resolve().parents[1] / "foundation/v1/gp_uncertainty_math.py"
spec = importlib.util.spec_from_file_location("gp_uncertainty_math", MODULE)
gp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gp)


class GPScalarMath(unittest.TestCase):
    def test_conditional_variance_bounds(self):
        rng = random.Random(20261027)
        for _ in range(100):
            prior = rng.uniform(.001, 10)
            observed = rng.uniform(.001, 10)
            cross = rng.uniform(-1, 1)*math.sqrt(prior*observed)
            cond = gp.conditioned_variance(prior, cross, observed)
            self.assertLessEqual(cond, prior+1e-12)
            self.assertGreaterEqual(cond, 0.)

    def test_disconnected_variance_unchanged(self):
        self.assertEqual(gp.conditioned_variance(3, 0, .1), 3.)

    def test_information_growth_and_zero_feature(self):
        self.assertEqual(gp.precision_increment(2., 0., .01), 2.)
        for noise in (.01, .1, 1.):
            self.assertGreater(gp.precision_increment(2., 3., noise), 2.)

    def test_invalid_covariance_and_input_rejection(self):
        for args in ((1., 2., 1.), (1., 0., 0.), (1., float('nan'), 1.)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                gp.conditioned_variance(*args)
        with self.assertRaises(ValueError):
            gp.precision_increment(1, 2, -1)


    def test_scalar_rankone_agrees_with_fraction_and_is_bounded(self):
        rng = random.Random(20261028)
        for _ in range(256):
            old = rng.uniform(0., 20.)
            feature = rng.uniform(-5., 5.)
            noise = rng.uniform(.05, 10.)
            actual = gp.scalar_rankone_variance(old, feature, noise)
            expected = old*noise/(noise+old*feature*feature)
            self.assertAlmostEqual(actual, expected, delta=1e-11)
            self.assertGreaterEqual(actual, 0.)
            self.assertLessEqual(actual, old + 1e-12)

    def test_rankone_rejects_invalid_premises(self):
        for values in ((-1., 1., 1.), (1., 1., 0.),
                       (1., float('inf'), 1.)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                gp.scalar_rankone_variance(*values)


if __name__ == "__main__":
    unittest.main()
