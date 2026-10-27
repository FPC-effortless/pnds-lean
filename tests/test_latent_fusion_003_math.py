import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"foundation"/"v1"))
from latent_fusion_003_math import contract,delta,delta_affine,replace

class FixedStateSubspaceMath(unittest.TestCase):
    def test_original_memory_capacity(self):
        c=contract()
        self.assertEqual(c["matrix_cells"],256)
        self.assertEqual(c["float_state_count"],1696)
        self.assertEqual(c["nominal_fp32_bytes"],6800)
    def test_inactive_head_identity(self):
        self.assertEqual(delta(1,0,2.4,.1,3.9),2.4)
    def test_real_scalar_affine_exactish(self):
        self.assertAlmostEqual(delta(.996,.18,2.4,.5,.7),
            delta_affine(.996,.18,2.4,.5,.7),places=14)
    def test_unselected_slots_untouched(self):
        old=[1.,2.,3.,4.]
        new=replace(old,[True,False,True,False],[10.,20.,30.,40.])
        self.assertEqual(new,[10.,2.,30.,4.])
    def test_difference_linear_operator(self):
        a,b,k,v=.997,.13,.66,.85
        left=delta(a,b,2.1,k,v)
        right=delta(a,b,1.6,k,v)
        self.assertAlmostEqual(left-right,a*(1-b*k*k)*(.5),places=13)
if __name__=="__main__":unittest.main()
