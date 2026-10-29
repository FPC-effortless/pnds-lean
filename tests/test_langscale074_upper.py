import importlib.util
import sys
import unittest
from pathlib import Path
p=Path(__file__).resolve().parents[1]/"foundation/v1/langscale074_upper.py"
spec=importlib.util.spec_from_file_location("langscale074_upper",p)
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)

class UpperMath(unittest.TestCase):
    def test_registered_geometry(self):
        for tier,(h,w,n) in m.TIERS.items():
            g=m.registered(tier)
            self.assertEqual(g.parameters,n)
            self.assertEqual(g.upper_state_width,h)
            self.assertEqual(g.ffn_width,w)
            self.assertEqual(g.upper_fp32_bytes_per_owner,4*h)
            self.assertEqual(g.native_associative_width,8)
            self.assertEqual(g.upper_recurrent_parameters,841+4*h*h+564*h)
    def test_parameter_parity(self):
        self.assertLess(m.live_gap(),0.01)
    def test_fail_closed(self):
        for p in ("h12","h160","",None):
            with self.assertRaises(ValueError):m.registered(p)
        for h,w in ((0,50),(-1,10),(20,0),(20,1.5),(True,20)):
            with self.assertRaises(ValueError):m.model_count(h,w)
