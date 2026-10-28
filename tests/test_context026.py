import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('context_ref',Path(__file__).resolve().parents[1]/'foundation/v1/context026.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
class ContextMath(unittest.TestCase):
    def test_oblique_interior_gate_counterexample(self):
        after,before=r.oblique_erase_counterexample()
        self.assertAlmostEqual(after,2.273832,places=12);self.assertEqual(before,2.)
        self.assertGreater(after,before)
    def test_no_admission_and_zero_key(self):
        m=[[1.,2.],[3.,4.]]
        self.assertEqual(r.native_step(m,[0.,0.],[1.,1.],.5,.7,False),m)
        self.assertEqual(r.native_step(m,[0.,0.],[1.,1.],.5,.7,True),[[.5,1.],[1.5,2.]])
    def test_envelope(self):
        force=8**.5;bound=200*force
        self.assertAlmostEqual(.995*bound+force,bound,places=10)
if __name__=='__main__':unittest.main()
