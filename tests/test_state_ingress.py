"""Partial NumPy scalar domain; not original full-model/IEEE equivalence."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

path=Path(__file__).resolve().parents[1]/'foundation/v1/state_ingress_20261110/reference.py'
spec=importlib.util.spec_from_file_location('ingress_ref',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Domain(unittest.TestCase):
    def test_valid_cone_and_exact_boundaries(self):
        for dtype in (np.float32,np.float64):
            d=np.array([0,1,2],dtype=dtype);n=np.array([-0.,1,-2],dtype=dtype)
            self.assertTrue(module.admissible(n,d,1e-8,dtype))
            self.assertTrue((np.abs(n/(d+1e-8))<=1).all())

    def test_wrong_mass_nonfinite_and_epsilon(self):
        for dtype in (np.float32,np.float64):
            for n,d in ((2,1),(0,-1),(np.nan,1),(0,np.inf)):
                self.assertFalse(module.admissible(np.array([n],dtype=dtype),np.array([d],dtype=dtype),1e-8,dtype))
        a=np.ones(2,dtype=np.float32)
        self.assertFalse(module.admissible(a,a,1e-100,np.float32))
        b=a.astype(np.float64)
        self.assertTrue(module.admissible(b,b,1e-100,np.float64))

    def test_randomized_predicate_and_bounded_read(self):
        gen=np.random.default_rng(17)
        for dtype in (np.float32,np.float64):
            d=gen.uniform(0,100,128).astype(dtype)
            n=(d*gen.uniform(-1,1,128)).astype(dtype)
            self.assertTrue(module.admissible(n,d,1e-8,dtype))
            self.assertTrue((np.abs(n/(d+1e-8))<=1).all())


if __name__=='__main__':unittest.main()
