import importlib.util,random,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('ref',Path(__file__).resolve().parents[1]/'foundation/v1/lacuna_survival001.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
class PartialReference(unittest.TestCase):
 def test_transport_additivity_random(self):
  rng=random.Random(633)
  for _ in range(100):
   a=[[rng.uniform(-1,1) for j in range(8)] for i in range(8)];b=[[rng.uniform(-1,1) for j in range(8)] for i in range(8)];k=[rng.uniform(-1,1) for j in range(8)]
   x=ref.transport([[a[i][j]+b[i][j] for j in range(8)] for i in range(8)],k,.997,.005);aa=ref.transport(a,k,.997,.005);bb=ref.transport(b,k,.997,.005)
   self.assertLess(max(abs(x[i][j]-aa[i][j]-bb[i][j]) for i in range(8) for j in range(8)),1e-14)
 def test_probability_mixture_and_budgets(self):
  p=[1/256]*256;r=[0.]*255+[1.]
  self.assertEqual(ref.probability_mixture(p,r,0),p)
  self.assertAlmostEqual(sum(ref.probability_mixture(p,r,.5)),1.)
  self.assertEqual(24+25*9+7,256);self.assertEqual(1920+8+1928+2,3858)
if __name__=='__main__':unittest.main()
