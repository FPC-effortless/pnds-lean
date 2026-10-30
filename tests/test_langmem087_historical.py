import importlib.util,random,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('reference087',ROOT/'foundation/v1/langmem087_historical.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
class HistoricalSplit(unittest.TestCase):
 def test_random_decomposition_and_zero_old(self):
  r=random.Random(87)
  for _ in range(50):
   m=[[r.uniform(-1,1) for _ in range(8)] for _ in range(8)];k=[r.uniform(-1,1) for _ in range(8)];v=[r.uniform(-1,1) for _ in range(8)]
   rho=.997;eta=.005;old,current=ref.split_slow(m,k,v,rho,eta)
   for i in range(8):
    for j in range(8):self.assertAlmostEqual(old[i][j]+current[i][j],rho*m[i][j]+eta*(v[i]-rho*sum(m[i][a]*k[a] for a in range(8)))*k[j],places=14)
   self.assertEqual(ref.attributed_read(m,k,v,k,rho,eta,'old_zero'),ref.attributed_read([[0.]*8 for _ in range(8)],k,v,k,rho,eta,'full'))
if __name__=='__main__':unittest.main()
