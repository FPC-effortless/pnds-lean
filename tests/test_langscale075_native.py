import importlib.util
import sys
import unittest
from pathlib import Path
p=Path(__file__).resolve().parents[1]/"foundation/v1/langscale075_native.py"
spec=importlib.util.spec_from_file_location("langscale075_native",p)
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)
class NativeGeometry(unittest.TestCase):
    def test_math(self):
        for tier,(k,n) in m.REGISTERED.items():
            g=m.registered(tier)
            self.assertEqual(g.active_parameters,n)
            self.assertEqual(g.native_active_matrix_shape,(k,k))
            self.assertEqual(g.aligned_each_shape,(16,k))
            self.assertEqual(g.native_active_fp32_bytes,4*k*k)
            self.assertEqual(g.aligned_two_fp32_bytes,128*k)
            self.assertEqual(g.total_three_bank_scan_fp32_bytes,192*k)
        self.assertLess(m.parity_fraction(),.01)
    def test_original_8(self):
        self.assertEqual(m.registered("m8").active_parameters,499623)
        self.assertEqual(m.registered("m8").native_active_fp32_bytes,256)
    def test_bad(self):
        for x in (True,12.0,-8,0,10):
            with self.assertRaises(ValueError):m.model_count(x)
        with self.assertRaises(ValueError):m.registered("m20")
