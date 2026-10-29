import importlib.util,random,unittest,sys
from pathlib import Path
p=Path(__file__).resolve().parents[1]/"foundation/v1/langmem078_dual.py"
s=importlib.util.spec_from_file_location("langmem078_dual",p)
m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
class DualMath(unittest.TestCase):
    def test_original_budget_and_rate(self):
        self.assertEqual(m.DUAL_PARAMETERS,499627)
        self.assertEqual(m.EXTRA_SLOW_FP32_BYTES,256)
        rho=m.rho_from_raw(-4.5)
        self.assertGreater(rho,.98)
        self.assertLess(rho,1)
    def test_contraction_and_input_boundedness(self):
        random.seed(7801)
        rho=m.rho_from_raw(-4.5)
        for _ in range(12):
            w=[random.uniform(-1.,1.) for _ in range(m.K)]
            norm=sum(z*z for z in w)**.5
            w=[x/norm for x in w]
            v=[random.uniform(-1.,1.) for _ in range(m.K)]
            s=[[0.]*m.K for _ in range(m.K)]
            for step in range(48):
                old=m.frobenius(s)
                s=m.slow_affine(s,w,v,rho,.013,.0027)
                self.assertLessEqual(m.frobenius(s),rho*old+.0027*m.K**.5+1e-12)
                self.assertLessEqual(m.frobenius(s),m.bound_after_steps(0.,rho,.0027,step+1)+1e-12)
            self.assertEqual(m.slow_affine(s,w,v,rho,.013,.0027,False),s)
    def test_fail_closed(self):
        with self.assertRaises(ValueError):m.rho_from_raw(-4.5,-1)
        s=[[0.]*8 for _ in range(8)]
        with self.assertRaises(ValueError):m.slow_affine(s,[1.]*8,[0.]*8,.9,.01,.002)
        with self.assertRaises(ValueError):m.slow_affine(s,[0.]*8,[0.]*8,.9,.1,.002)
