import importlib.util,math,random,sys,unittest
from pathlib import Path
p=Path(__file__).resolve().parents[1]/"foundation/v1/langmem077_retention.py"
spec=importlib.util.spec_from_file_location("langmem077_retention",p)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m
spec.loader.exec_module(m)
class NativeRetention(unittest.TestCase):
    def test_exact_source_same_inputs_and_gate_bound(self):
        rng=random.Random(7707)
        for k in (8,12,16):
            for _ in range(4):
                a=[[rng.uniform(-1,1) for _ in range(k)] for _ in range(k)]
                b=[[rng.uniform(-1,1) for _ in range(k)] for _ in range(k)]
                q=[[rng.uniform(-1,1) for _ in range(k)] for _ in range(k)]
                w=[rng.uniform(-1,1) for _ in range(k)]
                norm=math.sqrt(sum(z*z for z in w))
                w=[z/norm for z in w]
                alpha=rng.uniform(.05,.9);beta=rng.uniform(.02,.95)
                pmat=m.native_p(alpha,beta,w)
                from_sub=m.difference(m.step(a,pmat,q),m.step(b,pmat,q))
                analytic=m.right_product(m.difference(a,b),pmat)
                self.assertLess(m.frobenius(m.difference(from_sub,analytic)),1e-12)
                observed,bound=m.source_contraction_bound(m.difference(a,b),pmat,alpha)
                self.assertLessEqual(observed,bound+1e-12)
    def test_mask_and_source_proof_boundary(self):
        for wrong in (0,1,-1,10):
            with self.assertRaises(ValueError):m.native_p(.5,.5,[0.0]*wrong)
        with self.assertRaises(ValueError):m.native_p(1.1,.5,[0.0]*8)
        with self.assertRaises(ValueError):m.native_p(.5,2.0,[0.0]*8)
        with self.assertRaises(ValueError):m.native_p(.5,.5,[1.]*8)
