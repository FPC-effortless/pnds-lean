"""Partial independent objective/count reference; not whole model semantics."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

p=Path(__file__).resolve().parents[1]/'foundation/v1/byte100k_001/math_reference.py'
spec=importlib.util.spec_from_file_location('byte100k_reference',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ByteMathTest(unittest.TestCase):
    def test_count(self):self.assertEqual(m.parameter_count(),99050)
    def test_uniform_ce(self):
        loss,g=m.mean_byte_ce_and_logit_gradient(np.zeros((1,2,256)),np.array([[0,255]]))
        self.assertAlmostEqual(loss,np.log(256),places=14)
        self.assertTrue(np.allclose(g.sum(-1),0,atol=1e-14))
    def test_finite_difference(self):
        rng=np.random.default_rng(17);x=rng.normal(size=(1,3,256));y=np.array([[0,12,255]])
        _,g=m.mean_byte_ce_and_logit_gradient(x,y)
        a=x.copy();b=x.copy();a[0,1,12]+=1e-5;b[0,1,12]-=1e-5
        numeric=(m.mean_byte_ce_and_logit_gradient(a,y)[0]-m.mean_byte_ce_and_logit_gradient(b,y)[0])/2e-5
        self.assertAlmostEqual(numeric,g[0,1,12],places=9)
    def test_bad_target(self):
        with self.assertRaises(ValueError):m.mean_byte_ce_and_logit_gradient(np.zeros((1,1,256)),np.array([[256]]))

if __name__=='__main__':unittest.main()
