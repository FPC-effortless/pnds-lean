import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"foundation"/"v1"))
from latent_fusion_004_math import state_bytes,hard_edit,soft_edit

class FactorialKernelMath(unittest.TestCase):
    def test_two_head_full_matrix(self):
        self.assertEqual(state_bytes(2)["matrix"],256)
        self.assertEqual(state_bytes(2)["bytes"],6800)
    def test_four_head_full_matrix(self):
        self.assertEqual(state_bytes(4)["matrix"],256)
        self.assertEqual(state_bytes(4)["bytes"],6800)
    def test_hard_unselected(self):
        self.assertEqual(hard_edit(False,3.,9.),3.)
        self.assertEqual(hard_edit(True,3.,9.),9.)
    def test_soft_endpoints(self):
        self.assertAlmostEqual(soft_edit(0,3.,9.),3.)
        self.assertAlmostEqual(soft_edit(1,3.,9.),9.)
    def test_soft_sensitivity(self):
        p=.23
        self.assertAlmostEqual(soft_edit(p,3.,9.)-soft_edit(p,3.,7.),p*(9.-7.))
if __name__=="__main__":unittest.main()
