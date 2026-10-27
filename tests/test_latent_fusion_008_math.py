import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"foundation"/"v1"))
from latent_fusion_008_math import counts,fast_slow_current_outer,current_only_decode

class NoAssocControlMath(unittest.TestCase):
    def test_equal_99390_live_params(self):
        c=counts()
        self.assertEqual(c["associative_params"],99390)
        self.assertEqual(c["nonassociative_params"],99390)
    def test_identical_full_6800_stream(self):
        c=counts()
        self.assertEqual(c["matrix_elements"],256)
        self.assertEqual(c["fp32_values"],1696)
        self.assertEqual(c["per_stream_nominal_bytes"],6800)
    def test_current_cross_is_meaningful(self):
        self.assertEqual(fast_slow_current_outer([1,2],[3,4]),
                         [[3,4],[6,8]])
    def test_no_memory_argument_in_interface(self):
        seen=[]
        def f(x,clock):
            seen.append((x,clock))
            return x+clock
        self.assertEqual(current_only_decode(2,6,f),8)
        self.assertEqual(seen,[(2,6)])
if __name__=="__main__":unittest.main()
