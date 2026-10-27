"""Independent calendar/basis prepass and five-block balanced batch execution.

Uses only the independent NumPy model's scalar learned maps; no Torch or TAC
imports. All associative schedules are prepared with both stores erased.
"""
from dataclasses import replace
import numpy as np


def compose(a,b):
    pa,qa=a
    pb,qb=b
    return pa@pb,qa@pb+qb


def blelloch_prefix(p,q):
    """Source-shaped power-of-two padding, upsweep then exclusive downsweep."""
    b,t,h,k,_=p.shape
    padded=1 << (t-1).bit_length()
    identity=np.broadcast_to(np.eye(k,dtype=p.dtype),(b,h,k,k))
    zero=np.zeros_like(identity)
    leaves=[(p[:,i],q[:,i]) if i<t else (identity,zero) for i in range(padded)]
    def up(xs):
        if len(xs)==1: return xs[0],None,None
        m=len(xs)//2
        left,right=up(xs[:m]),up(xs[m:])
        return compose(left[0],right[0]),left,right
    out=[]
    def down(tree,incoming):
        total,left,right=tree
        if left is None:
            out.append(compose(incoming,total))
        else:
            down(left,incoming)
            down(right,compose(incoming,left[0]))
    down(up(leaves),(identity,zero))
    return np.stack([s[0] for s in out[:t]],1),np.stack([s[1] for s in out[:t]],1)


def prepass_and_scan(reference,data,query=None):
    obs,action,dt,admit,owners,ordinals=data
    s=reference.initial(owners)
    b,t,_=obs.shape
    contexts=[]
    ps,qs=[],[]
    identity=np.eye(8,dtype=obs.dtype)
    cat=lambda *xs:np.concatenate(xs,axis=-1)
    split=lambda x:np.split(x,2,axis=-1)
    from foundation.v1.source_aligned_model_math import normalized,sigmoid
    for i in range(t):
        # Context state carries no associative data at any prepass step.
        s=replace(s,memory=np.zeros_like(s.memory),aligned=np.zeros_like(s.aligned))
        contexts.append(s)
        _,after,trace=reference.step(s,obs[:,i],action[:,i],dt[:,i],admit[:,i],
                                     owners,ordinals[:,i],None if query is None else query[:,i])
        e=trace['embedding']
        ctx=s.fast_sum/max(s.count,1)
        zr=np.zeros((b,8),dtype=obs.dtype)
        p,fg=split(reference.core_dense('fast_map',cat(e,ctx,zr)))
        pf=ctx+sigmoid(fg)*(np.tanh(p)-ctx)
        d,g=split(reference.core_dense('slow_map',cat(pf,ctx,zr)))
        slow=ctx+.25*sigmoid(g)*(np.tanh(ctx+np.tanh(d))-ctx)
        a,beta=split(sigmoid(reference.core_dense('memory_gates',cat(e,pf,slow))))
        a=a*(1-beta/4)
        native_key=normalized(reference.core_dense('address',e))
        native_value=normalized(reference.core_dense('content',e))
        source_value=np.tanh(reference.dense('source_content',e))
        source_key=trace['aligned_write_key']
        # Block order is native, head0/half0, head0/half1, head1/half0, head1/half1.
        keys=np.stack((native_key,source_key[:,0],source_key[:,0],
                       source_key[:,1],source_key[:,1]),axis=1)
        values=np.stack((native_value,source_value[:,:8],source_value[:,8:],
                         source_value[:,:8],source_value[:,8:]),axis=1)
        alpha=np.concatenate((a,np.repeat(trace['aligned_retention'],2,axis=1)),axis=1)
        strength=np.concatenate((beta,np.repeat(trace['aligned_strength'],2,axis=1)),axis=1)
        p=alpha[...,None,None]*(identity-strength[...,None,None]*
                                keys[..., :,None]*keys[...,None,:])
        q=strength[...,None,None]*values[..., :,None]*keys[...,None,:]
        ps.append(np.where(admit[:,i,None,None,None],p,identity))
        qs.append(np.where(admit[:,i,None,None,None],q,0.))
        s=after
    p,q=np.stack(ps,1),np.stack(qs,1)
    _,scanned=blelloch_prefix(p,q)
    # Exact byte-preserving gather on source nonwrites, as in NUM-001.
    for i in range(t):
        previous=np.zeros_like(scanned[:,i]) if i==0 else scanned[:,i-1]
        scanned[:,i]=np.where(admit[:,i,None,None,None],scanned[:,i],previous)
    outputs=[]
    actual_states=[]
    for i in range(t):
        before=np.zeros_like(scanned[:,i]) if i==0 else scanned[:,i-1]
        restored=replace(contexts[i],memory=before[:,0],aligned=before[:,1:].reshape(b,2,16,8))
        y,state,_=reference.step(restored,obs[:,i],action[:,i],dt[:,i],admit[:,i],
                                  owners,ordinals[:,i],None if query is None else query[:,i])
        outputs.append(y)
        actual_states.append(state)
    return np.stack(outputs,1),scanned,actual_states
