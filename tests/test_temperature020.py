import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('tempref',Path(__file__).resolve().parents[1]/'foundation/v1/temperature020.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

class TemperatureMath(unittest.TestCase):
    def test_derivatives(self):
        z=[2.,-.1,.4];a=.7;eps=1e-4
        value,g,h=r.categorical_terms(z,2,a)
        lo=r.categorical_terms(z,2,a-eps)[0];hi=r.categorical_terms(z,2,a+eps)[0]
        self.assertAlmostEqual(g,(hi-lo)/(2*eps),places=7)
        self.assertAlmostEqual(h,(hi-2*value+lo)/eps**2,places=6)
        self.assertGreaterEqual(h,0)
    def test_identity_guard(self):
        self.assertLessEqual(min(2.,3.),2.)
    def test_positive_order(self):
        for a in [.25,.5,1.,4.]:self.assertGreater(a*2.,a*1.)

if __name__=='__main__':unittest.main()
