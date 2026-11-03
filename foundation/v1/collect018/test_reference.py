"""Execute the PNDS copy against actual source parameters; no model fitting."""
import copy,unittest
import torch
from collect018.common import build
from collect018.model import tensors
from .reference import whole_reference
class Paired(unittest.TestCase):
    def test_source_paired_fp32_fp64(self):
        torch.set_num_threads(1)
        for family in ('phase','context','gru'):
          for dtype in (torch.float32,torch.float64):
            with self.subTest(family=family,dtype=dtype):
                a=build(family+'_first',18000).to(dtype);b=copy.deepcopy(a);x=torch.arange(18).reshape(2,9)%256
                y,s=a(x);z,t=whole_reference(b,x);tol=4e-5 if dtype==torch.float32 else 3e-9
                torch.testing.assert_close(y,z,atol=tol,rtol=tol)
                for n,v in tensors(s).items():torch.testing.assert_close(v,tensors(t)[n],atol=tol,rtol=tol)
                y.square().sum().backward();z.square().sum().backward()
                for (n,p),(nn,q) in zip(a.named_parameters(),b.named_parameters()):
                    self.assertEqual(n,nn);torch.testing.assert_close(p.grad,q.grad,atol=tol*10,rtol=tol*10)
if __name__=='__main__':unittest.main()
