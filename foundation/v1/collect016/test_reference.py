"""Run with PYTHONPATH pointing to pinned research/collect016 and dependencies."""
import copy,importlib.util,json
from pathlib import Path
import torch
from collect016.common import build
from collect016.reference import whole_reference as source_reference
from collect011.model import named_state_tensors
spec=importlib.util.spec_from_file_location('pnds_c16_reference',Path(__file__).with_name('reference.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
def main():
    torch.set_num_threads(1);rows=[]
    for arm in ('full_bptt','detach_128','current_only'):
      for dtype in (torch.float32,torch.float64):
        m=build(arm,16000).to(dtype);n=copy.deepcopy(m);x=(torch.arange(18).reshape(2,9)*13)%256
        a,s=source_reference(m,x);b,t=module.whole_reference(n,x)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        for k,v in named_state_tensors(s).items():torch.testing.assert_close(v,named_state_tensors(t)[k],atol=0,rtol=0)
        a.square().mean().backward();b.square().mean().backward()
        for (k,v),(kk,w) in zip(m.named_parameters(),n.named_parameters()):
            assert k==kk;torch.testing.assert_close(v.grad,w.grad,atol=0,rtol=0)
        rows.append(dict(arm=arm,dtype=str(dtype),portable_reference_values_state_surrogate_gradients_exact=True))
    print(json.dumps(dict(status='PASS_PORTABLE_REFERENCE_CORRESPONDENCE_FIXTURES',rows=rows,optimizer_updates=0,full_Lean_IEEE_AD='OPEN'),indent=2))
if __name__=='__main__':main()
