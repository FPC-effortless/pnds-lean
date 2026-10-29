import importlib.util,unittest,random,sys,math
from pathlib import Path
path=Path(__file__).resolve().parents[1]/'foundation/v1/langmem085_selective_delta.py'
spec=importlib.util.spec_from_file_location('langmem085_delta_ref',path)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
class SourceDelta(unittest.TestCase):
    def test_source_registered_geometry(self):
        self.assertEqual(m.DIM,8)
    def test_full_latest_and_partial_residual(self):
        rng=random.Random(85001)
        for _ in range(35):
            k=[rng.uniform(-1,1) for _ in range(8)]
            kn=math.sqrt(sum(x*x for x in k));k=[x/kn for x in k]
            v=[rng.uniform(-1,1) for _ in range(8)]
            vn=math.sqrt(sum(x*x for x in v));v=[x/vn for x in v]
            M=[[rng.uniform(-.25,.25) for _ in range(8)] for _ in range(8)]
            rho=.989
            old=m.matvec(M,k)
            for eta in (1.,.25):
                after,res=m.delta(M,k,v,rho,eta)
                read=m.matvec(after,k)
                for i in range(8):
                    self.assertAlmostEqual(read[i],(1-eta)*rho*old[i]+eta*v[i],delta=3e-14)
                if eta==1.:
                    self.assertLess(math.dist(read,v),1e-12)
    def test_selective_and_clipped_gate_bounds(self):
        self.assertEqual(m.eta([0.]*8,'selective'),1.)
        self.assertEqual(m.eta([1.]*8,'selective'),.25)
        for x in (0.,.1,1.,9.):
            self.assertTrue(0<m.eta([x]+[0.]*7,'residual_clip035')<=1)
        with self.assertRaises(ValueError):m.eta([1.]*8,'unregistered')
