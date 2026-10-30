import random,unittest,math
from foundation.v1.langmem086_autonomous import clipped_update,residual_update,read,count_parameters,state_bytes
class AutonomousTests(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(count_parameters(),100521);self.assertEqual(state_bytes(),1920)
    def test_unit_read_and_clip_norm(self):
        r=random.Random(86)
        for _ in range(32):
            m=[[r.uniform(-.3,.3) for _ in range(8)] for _ in range(8)]
            key=[r.uniform(-1,1) for _ in range(8)];n=math.sqrt(sum(x*x for x in key));key=[x/n for x in key]
            value=[r.uniform(-1,1) for _ in range(8)];eta=.13;rho=.98
            updated=residual_update(m,key,value,rho,eta)
            for x,y,v in zip(read(updated,key),read(m,key),value):self.assertAlmostEqual(x,(1-eta)*rho*y+eta*v)
            clipped=clipped_update(m,key,value,rho,.7)
            norm=math.sqrt(sum((clipped[i][j]-rho*m[i][j])**2 for i in range(8) for j in range(8)))
            self.assertLessEqual(norm,.35*.7+1e-12)
if __name__=='__main__':unittest.main()
