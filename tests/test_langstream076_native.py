import importlib.util
import sys
import unittest
from pathlib import Path
p=Path(__file__).resolve().parents[1]/"foundation/v1/langstream076_native.py"
sp=importlib.util.spec_from_file_location("langstream076_native",p)
m=importlib.util.module_from_spec(sp)
sys.modules[sp.name]=m
sp.loader.exec_module(m)
class CarryMath(unittest.TestCase):
    def test_shapes(self):
        self.assertEqual([4*m.state_floats(k) for k in m.VALID],[3904,4752,5728])
    def test_mask_invariance_and_rank_one_example(self):
        k=8
        base=[[float(i==j) for j in range(k)] for i in range(k)]
        zero=[[0.0 for _ in range(k)] for _ in range(k)]
        p=[[0.5*float(i==j) for j in range(k)] for i in range(k)]
        self.assertEqual(m.native_write(base,p,zero,False),base)
        result=m.native_write(base,p,zero,True)
        self.assertEqual(result[0][0],.5)
        self.assertEqual(result[0][1],0)
    def test_fail_closed(self):
        for k in (9,True,16.0,-1):
            with self.assertRaises(ValueError):m.state_floats(k)
        x=[[0.0]*8 for _ in range(8)]
        with self.assertRaises(ValueError):m.native_write(x,x,x,"false")
