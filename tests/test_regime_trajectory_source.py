import os,sys,unittest
from pathlib import Path
import torch
from foundation.v1.regime_trajectory.reference import condition,moments,quantile,nll

class RegimeSource(unittest.TestCase):
    def test_complete_source_values_and_gradients(self):
        root=os.environ.get('TAC_SOURCE_ROOT')
        if not root:self.skipTest('source checkout required')
        sys.path.insert(0,str(Path(root)/'src'))
        from tac_osm.variants.regime_trajectory_world import RegimeTrajectoryWorld,regime_joint_nll,marginal_intervals
        torch.set_num_threads(1);torch.manual_seed(991)
        for dtype,rtol,atol in [(torch.float64,1e-8,1e-10),(torch.float32,1e-5,1e-6)]:
            m=RegimeTrajectoryWorld().to(dtype);x=torch.randn(2,5,14,dtype=dtype,requires_grad=True);a=torch.randn_like(x,requires_grad=True)
            valid=torch.ones(2,5,dtype=torch.bool)
            actual,state=m(x,a);par,logit,k,st=condition(dict(m.named_parameters())|dict(m.named_buffers()),x,a,valid)
            mean,scale=moments(par,(1,4,16));q=quantile(logit.sigmoid(),k)
            std=scale.diagonal(dim1=-2,dim2=-1).transpose(-1,-2).sqrt()
            lower,upper=mean-q[...,None]*std,mean+q[...,None]*std
            lo,hi=marginal_intervals(actual);truth=actual.mean.detach()+.015
            aloss=regime_joint_nll(actual,truth).mean()/42+(hi-lo).mean()+actual.mean.square().mean()
            rloss=nll(mean,scale,logit,k,truth).mean()/42+(upper-lower).mean()+mean.square().mean()
            for z,w in [(actual.mean,mean),(actual.narrow_scale_matrix,scale),(lo,lower),(hi,upper),(aloss,rloss)]:torch.testing.assert_close(z,w,rtol=rtol,atol=atol)
            params=tuple(m.parameters())+(x,a)
            g=torch.autograd.grad(aloss,params,retain_graph=True);gr=torch.autograd.grad(rloss,params)
            for z,w in zip(g,gr):torch.testing.assert_close(z,w,rtol=rtol,atol=atol)


    def test_masked_padding_correspondence(self):
        root=os.environ.get('TAC_SOURCE_ROOT')
        if not root:self.skipTest('source checkout required')
        sys.path.insert(0,str(Path(root)/'src'))
        from tac_osm.variants.regime_trajectory_world import RegimeTrajectoryWorld
        torch.manual_seed(992);m=RegimeTrajectoryWorld().double()
        x=torch.randn(2,9,14,dtype=torch.float64);a=torch.randn_like(x)
        valid=torch.ones(2,9,dtype=torch.bool);valid[0,2:4]=False
        actual,st=m(x,a,valid=valid)
        par,l,k,refstate=condition(dict(m.named_parameters())|dict(m.named_buffers()),x,a,valid)
        mean,scale=moments(par,(1,4,16),valid=valid)
        for z,w in [(actual.mean,mean),(actual.narrow_scale_matrix,scale),(st.basis_sum,refstate[0]),(st.memory,refstate[2])]:
            torch.testing.assert_close(z,w,rtol=1e-8,atol=1e-10)
