"""Independent serial full neural equation reference, not TAC source imports.

Uses the same parameter tensors, PyTorch elementary nonlinearities and FP
arithmetic, but independent chronological rank-one memory update; no scan.
"""
import torch
from torch.nn import functional as F

def condition(weights, obs, action, valid, initial=None):
    B,T,_=obs.shape; width=weights['fast_map.bias'].numel()//2
    def lin(x,p):return F.linear(x,weights[p+'.weight'],weights[p+'.bias'])
    if initial is None:
        total=obs.new_zeros(B,width); count=torch.zeros(B,1,dtype=torch.int64,device=obs.device)
        memory=obs.new_zeros(B,8,8)
    else: total,count,memory=initial
    outputs=[]
    for t in range(T):
        x,u,mask=obs[:,t],action[:,t],valid[:,t,None]
        e=lin(torch.cat((x,u),-1),'encoder').tanh()
        k=F.normalize(lin(e,'address'),dim=-1,eps=1e-6)
        v=F.normalize(lin(e,'content'),dim=-1,eps=1e-6)
        zero=e.new_zeros(B,8)
        basis=lin(torch.cat((e,e.new_zeros(B,width),zero),-1),'fast_map').chunk(2,-1)[0].tanh()
        ctx=total/count.clamp_min(1).to(obs.dtype)
        p,g=lin(torch.cat((e,ctx,zero),-1),'fast_map').chunk(2,-1)
        pf=(1-g.sigmoid())*ctx+g.sigmoid()*p.tanh()
        d,g=lin(torch.cat((pf,ctx,zero),-1),'slow_map').chunk(2,-1)
        ps=(1-.25*g.sigmoid())*ctx+.25*g.sigmoid()*(ctx+d.tanh()).tanh()
        raw_a,raw_b=lin(torch.cat((e,pf,ps),-1),'memory_gates.inner_linear').chunk(2,-1)
        # Match the actual controlled-logit wrapper's finite precision stages.
        aa=(1-.02*(1-raw_a.sigmoid())).clamp(max=1-1e-5)
        bb=.25*raw_b.sigmoid()
        alpha=torch.logit(aa).sigmoid()*(1-torch.logit(bb).sigmoid()/4)
        beta=torch.logit(bb).sigmoid()
        q=F.normalize(k+lin(torch.cat((e,pf,ps),-1),'query'),dim=-1,eps=1e-6)
        r=(memory@q[...,None]).squeeze(-1)
        f,g=lin(torch.cat((e,pf,r),-1),'fast_map').chunk(2,-1)
        f=(1-g.sigmoid())*pf+g.sigmoid()*f.tanh()
        d,g=lin(torch.cat((f,ps,r),-1),'slow_map').chunk(2,-1)
        s=(1-.25*g.sigmoid())*ps+.25*g.sigmoid()*(ps+d.tanh()).tanh()
        raw=lin(F.silu(lin(torch.cat((f,s,r,e),-1),'readout.0')),'readout.2')
        raw=raw+lin(F.silu(lin(r,'retrieved_delta.0')),'retrieved_delta.2')
        hidden=F.silu(lin(torch.cat((f,s,r,e),-1),'readout.0'))
        vi,rr,sx,sv=lin(hidden,'trajectory_head').chunk(4,-1)
        amplitude=weights['delta_scale']*(.02+F.softplus(sx))
        share=.05+.90*sv.sigmoid()
        outputs.append((x,weights['delta_scale']*raw.tanh(),weights['delta_scale']*vi.tanh(),
                        .01+.98*rr.sigmoid(),amplitude*share.sqrt(),amplitude*(1-share).sqrt()))
        oldread=(memory@k[...,None]).squeeze(-1)
        written=alpha[...,None]*memory+(beta*(v-alpha*oldread))[...,None]*k[:,None,:]
        memory=torch.where(mask[...,None],written,memory)
        total=total+torch.where(mask,basis,torch.zeros_like(basis))
        count=count+mask.to(torch.int64)
    return tuple(torch.stack([o[j] for o in outputs],1) for j in range(6)),(total,count,memory)


def moments(parameters,horizons):
    """Independent finite state-power/noise-response reference, no closed division.

    This explicitly propagates sensitivities to independent future innovations.
    Gaussian scale mixing into df4 Student-t is specified outside this kernel.
    """
    x,v,vi,r,sx,sv=parameters
    nmax=max(horizons);means={0:x};position=x;velocity=v
    # Each basis slot corresponds to one future position/velocity innovation.
    px=torch.zeros(*x.shape,nmax,device=x.device,dtype=torch.float64);pv=torch.zeros_like(px)
    vx=torch.zeros_like(px);vv=torch.zeros_like(px)
    response={0:(px,pv)}
    for j in range(nmax):
        eye=x.new_zeros(nmax);eye[j]=1
        px=px+vx+sx.double()[...,None]*eye
        pv=pv+vv
        vx=r.double()[...,None]*vx
        vv=r.double()[...,None]*vv+sv.double()[...,None]*eye
        position=position+velocity
        velocity=r*velocity+(1-r)*vi
        means[j+1]=position;response[j+1]=(px,pv)
    mu=torch.stack([means[h] for h in horizons],-2)
    rows=[]
    for h in horizons:
        row=[]
        for k in horizons:
            ax,av=response[h];bx,bv=response[k]
            row.append((ax*bx+av*bv).sum(-1))
        rows.append(torch.stack(row,-1))
    scale=torch.stack(rows,-2)
    return mu,scale
