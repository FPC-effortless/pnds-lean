"""LANGHEAD094 diagnostic exact-real Python recurrence checks."""
import math
import random
import unittest
from foundation.v1.langhead094_math import diagonal_scalar,diagonal_vector,gru_blend,gru_scalar
class LANGHEAD094MathTest(unittest.TestCase):
    def test_ssm_state_dependency_is_contracting(self):
        for seed in (1,23,49):
            rng=random.Random(seed)
            for _ in range(50):
                v,w,u=(rng.uniform(-2,2) for j in range(3))
                rate=rng.uniform(-4,4);gate=rng.uniform(-4,4)
                x=diagonal_scalar(v,u,rate,gate)
                y=diagonal_scalar(w,u,rate,gate)
                self.assertLessEqual(abs(x-y),abs(v-w)+1e-10)
    def test_ssm_candidate_fixed_point(self):
        for offer in (-2.,0.,3.):
            self.assertAlmostEqual(diagonal_scalar(offer,offer,0.,.5),offer,places=12)
    def test_vector_matches_scalar(self):
        a=diagonal_vector((1.,2.,3.),(-1.,0.,4.),(0.,1.,2.),(.5,1.5,3.))
        self.assertEqual(len(a),3)
        for i in range(3):self.assertAlmostEqual(a[i],diagonal_scalar((1.,2.,3.)[i],(-1.,0.,4.)[i],(0.,1.,2.)[i],(.5,1.5,3.)[i]))
    def test_gru_interpolation_contract(self):
        for z in (0.,.1,.9,1.):
            self.assertAlmostEqual(gru_blend(5.,5.,z),5.)
            self.assertLessEqual(abs(gru_blend(-2.,1.,z)-gru_blend(3.,1.,z)),5.+1e-10)
    def test_gru_scalar_finite(self):
        z=gru_scalar(1.,.2,.5,.1,-.1,.3,.4)
        self.assertTrue(math.isfinite(z))
    def test_invalid_guards(self):
        with self.assertRaises(ValueError):diagonal_vector((1.,),(1.,2.),(1.,),(1.,))
        with self.assertRaises(ValueError):diagonal_scalar(0.,0.,float('nan'),0.)
        with self.assertRaises(ValueError):gru_blend(0.,0.,2.)
if __name__=='__main__':unittest.main()
