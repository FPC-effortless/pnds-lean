"""Independent full-source reference; old serial core plus global regime."""
import torch
from torch.nn import functional as F
from foundation.v1.coherent_trajectory.reference import condition as base_condition,moments as base_moments

def condition(weights,obs,action,valid,initial=None):
    params,state=base_condition(weights,obs,action,valid,initial)
    x,v,u,r,sx,sv=params;scale=weights['delta_scale']
    feature=torch.cat((v/scale,u/scale,r,((sx.square()+sv.square()).sqrt()/scale).log()),-1)
    logit=F.linear(feature,weights['regime_gate.weight'],weights['regime_gate.bias'])
    k=1+F.softplus(weights['raw_broadening'])
    return params,logit,k,state

def cdf(z):
    u=torch.sin(torch.atan(z/2))
    return .5+(3*u-u.pow(3))/4

def pdf(z):return .375*(1+z.square()/4).pow(-2.5)

def quantile(p,k):
    p=p.double();k=k.double()
    with torch.no_grad():
        lo=torch.zeros_like(p);hi=torch.full_like(p,2.13184678632665)*k
        for _ in range(80):
            mid=(lo+hi)/2;f=(1-p)*cdf(mid)+p*cdf(mid/k)
            lo=torch.where(f<.95,mid,lo);hi=torch.where(f>=.95,mid,hi)
        q=(lo+hi)/2
        f=(1-p)*pdf(q)+p*pdf(q/k)/k
        dp=(cdf(q)-cdf(q/k))/f
        dk=p*pdf(q/k)*q/k.square()/f
    return q+(p-p.detach())*dp+(k-k.detach())*dk

def nll(mean,scale,logit,k,truth):
    r=(truth-mean).transpose(-1,-2).double();H=r.shape[-1]
    def component(s):
        q=(r[...,None,:]@torch.linalg.solve(s,r[...,None])).squeeze(-1).squeeze(-1)
        _,det=torch.linalg.slogdet(s)
        c=torch.lgamma(s.new_tensor((4+H)/2))-torch.lgamma(s.new_tensor(2.))-.5*H*torch.log(s.new_tensor(4*torch.pi))
        return (c-.5*det-(4+H)/2*torch.log1p(q/4)).sum(-1)
    a=component(scale);b=component(scale*k.double().square())
    l=logit.squeeze(-1).double()
    return -torch.logaddexp(F.logsigmoid(-l)+a,F.logsigmoid(l)+b)


def moments(parameters,horizons,valid=None):
    mean,scale=base_moments(parameters,horizons)
    if valid is not None:
        mean=torch.where(valid[...,None,None],mean,parameters[0][...,None,:].expand_as(mean))
        scale=torch.where(valid[...,None,None,None],scale,torch.zeros_like(scale))
    return mean,scale
