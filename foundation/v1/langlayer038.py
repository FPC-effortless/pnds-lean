"""Independent serial upper038 graph; not the composed full TAC/IEEE/AD proof."""
import torch
from torch.nn import functional as F


def serial_upper(weights,z):
    if z.ndim!=3 or z.shape[-1]!=280 or z.shape[1]==0: raise ValueError('expected nonempty B,T,280')
    def linear(name,x): return F.linear(x,weights[name+'.weight'],weights.get(name+'.bias'))
    state=z.new_zeros((z.size(0),weights['input_proj.weight'].size(0)))
    outputs=[];history={name:[] for name in ('state','prior_state','read_gate','alpha','offer')}
    for t in range(z.size(1)):
        old=state
        norm=F.layer_norm(z[:,t],(280,),weights['input_norm.weight'],weights['input_norm.bias'],1e-5)
        u=linear('input_proj',norm).tanh()
        retain=.995*linear('forget',u).sigmoid();offer=linear('offer',u).tanh()
        gate=linear('read_gate',torch.cat((u,old),-1)).sigmoid()
        outputs.append(z[:,t]+weights['residual_gain'].sigmoid()*linear('output',(u+gate*old).tanh()))
        state=retain*old+(1-retain)*offer
        for name,value in [('state',state),('prior_state',old),('read_gate',gate),('alpha',retain),('offer',offer)]:
            history[name].append(value)
    return torch.stack(outputs,1),{**{name:torch.stack(values,1) for name,values in history.items()},'gain':weights['residual_gain'].sigmoid()}
