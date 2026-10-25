"""Explicit actual-source numerical/gradient parity, optional torch dependency."""
import os,sys,unittest
from pathlib import Path
import torch
from foundation.v1.evidence_retrieval.reference import evaluate

class SourceParity(unittest.TestCase):
    def test_original_source(self):
        root=os.environ.get('TAC_SOURCE_ROOT')
        if not root:self.skipTest('TAC_SOURCE_ROOT required; no source-equivalence claim without it')
        sys.path.insert(0,str(Path(root)/'src'))
        from tac_osm.variants.evidence_retrieval_world import EvidenceRetrievalWorld
        torch.set_num_threads(1);torch.manual_seed(212)
        for dtype,rtol,atol in [(torch.float64,1e-8,1e-10),(torch.float32,1e-5,1e-6)]:
            m=EvidenceRetrievalWorld().to(dtype)
            x=torch.randn(2,17,14,dtype=dtype,requires_grad=True)
            u=torch.randn_like(x,requires_grad=True)
            mask=torch.rand(2,17)>.2
            y,s=m(x,u,valid=mask,return_state=True)
            z,rs=evaluate(dict(m.named_parameters())|dict(m.named_buffers()),x,u,mask)
            torch.testing.assert_close(y,z,rtol=rtol,atol=atol)
            for a,b in zip((s.basis_sum,s.count,s.memory),rs):
                torch.testing.assert_close(a,b,rtol=rtol,atol=atol)
            params=tuple(m.parameters())+(x,u)
            a=torch.autograd.grad(y.square().sum()+s.memory.square().sum(),params,retain_graph=True)
            b=torch.autograd.grad(z.square().sum()+rs[2].square().sum(),params)
            for ga,gb in zip(a,b):torch.testing.assert_close(ga,gb,rtol=rtol,atol=atol)
