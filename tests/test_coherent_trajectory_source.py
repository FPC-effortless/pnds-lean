import os,sys,unittest
from pathlib import Path
import torch
from foundation.v1.coherent_trajectory.reference import condition,moments

class CoherentSource(unittest.TestCase):
    def test_all_maps_and_gradients(self):
        root=os.environ.get('TAC_SOURCE_ROOT')
        if not root:self.skipTest('actual TAC source checkout required')
        sys.path.insert(0,str(Path(root)/'src'))
        from tac_osm.variants.coherent_trajectory_world import CoherentTrajectoryWorld
        torch.set_num_threads(1);torch.manual_seed(981)
        for dtype,rtol,atol in [(torch.float64,1e-8,1e-10),(torch.float32,1e-5,1e-6)]:
            m=CoherentTrajectoryWorld().to(dtype);x=torch.randn(2,7,14,dtype=dtype,requires_grad=True);a=torch.randn_like(x,requires_grad=True)
            valid=torch.ones(2,7,dtype=torch.bool)
            out,state=m(x,a)
            par,refstate=condition(dict(m.named_parameters())|dict(m.named_buffers()),x,a,valid)
            mean,scale=moments(par,(1,4,16))
            torch.testing.assert_close(out.mean,mean,rtol=rtol,atol=atol)
            torch.testing.assert_close(out.scale_matrix,scale,rtol=rtol,atol=atol)
            for z,w in zip((state.basis_sum,state.count,state.memory),refstate):torch.testing.assert_close(z,w,rtol=rtol,atol=atol)
            params=tuple(m.parameters())+(x,a)
            g=torch.autograd.grad(out.mean.square().mean()+out.scale_matrix.square().mean(),params,retain_graph=True)
            gr=torch.autograd.grad(mean.square().mean()+scale.square().mean(),params)
            for z,w in zip(g,gr):torch.testing.assert_close(z,w,rtol=rtol,atol=atol)
