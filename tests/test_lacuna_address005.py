import importlib.util,random,unittest
from pathlib import Path
s=importlib.util.spec_from_file_location('reference005',Path(__file__).resolve().parents[1]/'foundation/v1/lacuna_address005.py');ref=importlib.util.module_from_spec(s);s.loader.exec_module(ref)
class SpanReference(unittest.TestCase):
    def test_three_norms(self):
        rng=random.Random(5005)
        for _ in range(100):
            v,h,c,f=([rng.uniform(-1,1) for _ in range(8)] for _ in range(4))
            out,status,rank=ref.project(v,h,c,f)
            self.assertEqual((status,rank),('REGULAR',2))
            self.assertAlmostEqual(ref.norm(out),ref.norm(h),places=12)
            self.assertAlmostEqual(ref.norm([x+y for x,y in zip(out,c)]),ref.norm([x+y for x,y in zip(h,c)]),places=12)
            self.assertAlmostEqual(ref.norm([.7*a+.3*(b+d) for a,b,d in zip(f,out,c)]),ref.norm([.7*a+.3*(b+d) for a,b,d in zip(f,h,c)]),places=12)
    def test_dependent_axes(self):
        h=[1.,1.]+[0.]*6;c=[1.]+[0.]*7
        out,status,rank=ref.project([0.,0.,1.]+[0.]*5,h,c,c)
        self.assertEqual((status,rank),('REGULAR',1));self.assertAlmostEqual(ref.norm(out),ref.norm(h))
if __name__=='__main__':unittest.main()
