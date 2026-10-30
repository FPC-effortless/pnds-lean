import unittest,importlib.util,sys,math,random
from pathlib import Path
p=Path(__file__).resolve().parents[1]/"foundation/v1/langmem088_slow_bias.py"
spec=importlib.util.spec_from_file_location("langmem088_slow_bias",p)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
class FrozenSlowReadBias(unittest.TestCase):
    def test_parameters_and_model_fp32_per_owner_unchanged(self):
        self.assertEqual(m.TRAINABLE,100521)
        self.assertEqual(m.PERSISTENT_FP32_BYTES,1920)
        self.assertEqual(m.apply_bias([.2,-.3]),(.2,.7))
    def test_plus_one_slow_preference_monotone_without_predictive_claim(self):
        rng=random.Random(8809)
        for _ in range(64):
            f=rng.uniform(-8,8);s=rng.uniform(-8,8)
            p=m.slow_share(f,s)
            q=m.slow_share(f,s,1.)
            self.assertGreater(q,p)
            self.assertGreaterEqual(m.slow_share_derivative(f,s),0.)
    def test_analytic_gradient_matches_finite_difference(self):
        for f,s in [(-2.,0.),(0.,0.),(.5,1.2),(3.,-.6)]:
            h=1e-5
            numerical=(m.slow_share(f,s,h)-m.slow_share(f,s,-h))/(2*h)
            self.assertAlmostEqual(numerical,m.slow_share_derivative(f,s),delta=1e-9)
    def test_not_a_categorical_retrieval_or_parameter_freezing_proof(self):
        with self.assertRaises(ValueError):m.apply_bias([.2])
