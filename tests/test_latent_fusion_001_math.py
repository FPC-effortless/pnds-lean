"""Only scalar operator engineering checks; no trained-model verification."""
import math
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "foundation" / "v1"))
from latent_fusion_001_math import (affine_apply, affine_compose,
  decay_impute, gated_delta, source_slot_write)


class LatentFusion001ReferenceTests(unittest.TestCase):
    def test_affine_chronological_composition(self):
        first, second = (.4, 2.), (.6, -1.)
        a, b = affine_compose(first, second)
        self.assertAlmostEqual(affine_apply(a, b, 3.),
                               affine_apply(*second, affine_apply(*first, 3.)))

    def test_imputation_missingness(self):
        self.assertEqual(decay_impute(5., True, 2., 4., 0., 2.), 5.)
        self.assertAlmostEqual(decay_impute(float("nan"), False, 2., 4., 0., .5),
                               2 * math.exp(-2))
        with self.assertRaises(ValueError):
            decay_impute(0., True, 0., -1., 0., 1.)

    def test_gated_delta_one_hot_normalized_exact_target(self):
        k = [1., 0.]
        result = gated_delta([[0.,0.],[0.,0.]], k, [2., -1.], 1., 1.)
        self.assertAlmostEqual(result[0][0], 2.)
        self.assertAlmostEqual(result[1][0], -1.)
        self.assertEqual(result[0][1], 0.)
        with self.assertRaises(ValueError):
            gated_delta([[0., 0.]], k, [1.], -1., 1.)

    def test_slot_isolation(self):
        old = [[1., 2.], [3., 4.], [5., 6.]]
        new, active = source_slot_write(old, [False]*3, 1, [9., 10.], .7)
        self.assertEqual(new[0], old[0])
        self.assertEqual(new[2], old[2])
        self.assertTrue(active[1])
        with self.assertRaises(ValueError):
            source_slot_write(old, [False]*3, 3, [2., 3.], .5)


if __name__ == "__main__":
    unittest.main()
