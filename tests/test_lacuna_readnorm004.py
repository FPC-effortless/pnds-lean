import importlib.util,random,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('readnorm',Path(__file__).resolve().parents[1]/'foundation/v1/lacuna_readnorm004.py')
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
class ReadNormReference(unittest.TestCase):
    def test_random_constraints(self):
        rng=random.Random(4)
        for _ in range(100):
            v,h,c=([rng.uniform(-1,1) for _ in range(8)] for _ in range(3))
            out,status=ref.dual_norm(v,h,c)
            self.assertEqual(status,'REGULAR')
            self.assertAlmostEqual(ref.dot(out,out),ref.dot(h,h),places=12)
            self.assertAlmostEqual(ref.dot(out,c),ref.dot(h,c),places=12)
            self.assertAlmostEqual(ref.norm([a+b for a,b in zip(out,c)]),ref.norm([a+b for a,b in zip(h,c)]),places=12)
    def test_degenerate_original_fallback(self):
        h=[1.]+[0.]*7;c=h[:]
        self.assertEqual(ref.dual_norm([0.,1.]+[0.]*6,h,c),(h,'NO_ORTHOGONAL_FREEDOM'))
        self.assertEqual(ref.dual_norm([0.]*8,h,[0.]*8),(h,'INCONCLUSIVE_ZERO_CANDIDATE'))
if __name__=='__main__':unittest.main()
