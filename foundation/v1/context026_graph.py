"""Independent chronological CONTEXT026 full forward/VJP reference.

No TAC imports, source forward calls, scan helpers or source traces. Shares only
explicit parameter tensors and PyTorch primitives, so this is NOT an independent
AD implementation or a formal IEEE error proof. Reset windows only.
"""
import math
import torch
from torch.nn import functional as F


def serial_graph(weights, tokens, write_mask=None):
    if tokens.ndim != 2 or tokens.dtype != torch.long or tokens.shape[1] == 0:
        raise ValueError('expected nonempty B,T int64 tokens')
    if tokens.min() < 0 or tokens.max() >= weights['embedding.weight'].shape[0]:
        raise ValueError('token outside vocabulary')
    b, length = tokens.shape
    dtype, device = weights['embedding.weight'].dtype, tokens.device
    if write_mask is None:
        write_mask = torch.ones_like(tokens, dtype=torch.bool)
    if write_mask.shape != tokens.shape or write_mask.dtype != torch.bool:
        raise ValueError('invalid admission mask')
    def zeros(*dims): return torch.zeros(*dims, dtype=dtype, device=device)
    def linear(name, x): return F.linear(x, weights[name+'.weight'], weights.get(name+'.bias'))
    def unit(x): return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    def ordinal(t):
        parts=[]
        for period in (7,13,29,59):
            # Keep input dtype arithmetic explicit; source uses remainder then cast.
            a=torch.tensor(t % period, dtype=dtype, device=device)*(2*math.pi/period)
            parts.extend((a.cos(),a.sin()))
        return torch.stack(parts).expand(b,-1)
    native=zeros(b,8,8); aligned=zeros(b,2,16,8)
    wave=zeros(b,8,2); concepts=zeros(b,6,16); total_basis=zeros(b,88)
    lexicals=[]; keys=[]; outputs=[]; traces={}
    def record(**values):
        for name,value in values.items(): traces.setdefault(name,[]).append(value)
    eye6=torch.eye(6,dtype=dtype,device=device)
    angle=weights['wave_angle']
    rotation=torch.stack((torch.stack((angle.cos(),-angle.sin()),-1),
                          torch.stack((angle.sin(),angle.cos()),-1)),-2)
    for t in range(length):
        ti=torch.tensor(t,dtype=dtype,device=device)
        position=torch.stack((torch.log1p(ti),torch.sin(ti/16),torch.cos(ti/16),
            torch.sin(ti/64),torch.cos(ti/64),torch.sin(ti/256),torch.cos(ti/256)))
        embed=F.embedding(tokens[:,t],weights['embedding.weight'])
        e=torch.tanh(embed+linear('pos',position))
        lexical=torch.tanh(embed); local=lexical
        for lag in range(1,4):
            if lag <= t: local=local+weights['context_lag_weights'][lag-1].tanh()*lexicals[t-lag]
        key=unit(linear('native_key',local.tanh()))
        write_key=keys[-1] if t else zeros(b,8)
        value=linear('native_value',e).tanh()
        ab=linear('native_gate',e).sigmoid(); alpha=.995*ab[:,0]; beta=ab[:,1]
        admitted=write_mask[:,t] & (t>0)
        native_before=native
        retained=alpha[:,None,None]*native
        residual=value-(retained@write_key[:,:,None]).squeeze(-1)
        proposed=retained+beta[:,None,None]*residual[:,:,None]*write_key[:,None,:]
        native=torch.where(admitted[:,None,None],proposed,native)
        read=(native@key[:,:,None]).squeeze(-1)
        akey=unit(torch.einsum('bi,hij->bhj',ordinal(t),weights['rotation']))
        previous_key=unit(torch.einsum('bi,hij->bhj',ordinal(max(0,t-1)),weights['rotation']))
        aligned_read=(aligned@previous_key[:,:,:,None]).squeeze(-1).mean(1) if t else zeros(b,16)
        ag=linear('aligned_gate',e).sigmoid().reshape(b,2,3)
        av=linear('aligned_value',e).tanh()
        aligned_before=aligned
        retained_aligned=ag[:,:,0,None,None]*aligned
        old_projection=(retained_aligned@akey[:,:,:,None]).squeeze(-1)
        aligned_offer=retained_aligned-ag[:,:,1,None,None]*old_projection[:,:,:,None]*akey[:,:,None,:]
        aligned_offer=aligned_offer+ag[:,:,2,None,None]*av[:,None,:,None]*akey[:,:,None,:]
        aligned=torch.where(write_mask[:,t,None,None,None],aligned_offer,aligned)
        oldwave=wave.flatten(1); oldcp=concepts
        ctx=total_basis/max(1,t)
        fast=linear('fast',torch.cat((e,ctx,oldwave,read),-1)).tanh()
        slow=linear('slow',torch.cat((fast,ctx,read),-1)).tanh()
        evidence=linear('evidence',torch.cat((e,fast,slow,aligned_read,oldwave),-1)).tanh().reshape(b,6,16)
        query=linear('concept_q',evidence); ckey=linear('concept_k',evidence)
        sm=(query@ckey.transpose(-1,-2)/4).softmax(-1)
        adjacency=.5*(sm+sm.transpose(-1,-2))*(1-eye6)
        laplacian=torch.diag_embed(adjacency.sum(-1))-adjacency
        posterior=torch.linalg.solve(1.25*eye6+.25*laplacian,evidence+.25*oldcp)
        pooled=(posterior*evidence.mean(-1).softmax(-1)[:,:,None]).sum(1)
        z=F.layer_norm(torch.cat((fast,slow,read,aligned_read,oldwave,pooled,e),-1),
            (280,),weights['layernorm.weight'],weights['layernorm.bias'],eps=1e-5)
        logits=linear('ffn_out',linear('ffn_up',z)*F.silu(linear('ffn_gate',z)))+linear('concept_out',pooled)
        outputs.append(logits)
        record(native_prewrite=native_before,native_memory=native,native_read=read,
            native_query_key=key,native_write_key=write_key,native_value=value,native_alpha=alpha,
            native_beta=beta,native_admission=admitted,aligned_prewrite=aligned_before,
            wave_prewrite=oldwave,concept_prewrite=oldcp,concept_posterior=posterior,
            concept_laplacian=laplacian,gate_beta=torch.cat((beta[:,None],ag[:,:,2]),-1),basis=ctx)
        # Updates are chronological; all sources through current observed byte.
        rho=.97+.029*linear('wave_decay',e).sigmoid()
        wave=rho[:,:,None]*(wave[:, :,None,:]@rotation).squeeze(-2)+.025*linear('wave_input',e).tanh().reshape(b,8,2)
        cg=.05*linear('concept_gate',e).sigmoid().reshape(b,6,16)
        concepts=(1-cg)*concepts+cg*linear('concept_value',e).tanh().reshape(b,6,16)
        total_basis=total_basis+linear('basis',e).tanh()
        lexicals.append(lexical); keys.append(key)
    return torch.stack(outputs,1),{name:torch.stack(values,1) for name,values in traces.items()}
