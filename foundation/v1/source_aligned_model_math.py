"""Independent NumPy equations for the complete PAR-MEM-010 forward graph.

Source pin: TAC-OSM 3b278acc9daf587fd20892e470c857bfc933248e.
No imports from Torch or TAC. No optimizer, dataset or forecasting claims.
The sequential reference intentionally does not reproduce the source scan.
NumPy execution is numerical evidence, not an IEEE refinement proof.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

SOURCE_COMMIT = "3b278acc9daf587fd20892e470c857bfc933248e"
PERIODS = (7, 13, 29, 59)


def sigmoid(x):
    # Stable at large finite inputs; exact-real definition is 1/(1+exp(-x)).
    return np.exp(-np.logaddexp(0, -np.asarray(x)))


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def normalized(x):
    return x / np.maximum(np.sqrt((x*x).sum(axis=-1, keepdims=True)), 1e-6)


def ordinal_features(n, dtype):
    if n.dtype != np.int64 or np.any(n < 0):
        raise ValueError("nonnegative int64 ordinals required")
    return np.stack([f((n % p).astype(dtype)*(2*np.pi/p))
                     for p in PERIODS for f in (np.cos, np.sin)], axis=-1)


def corrective_write(memory, key, value, alpha, beta, admit):
    """[B,H,V,K], [B,H,K], [B,V], [B,H], [B,H], [B]."""
    predicted = np.sum(memory * key[..., None, :], axis=-1)
    residual = value[:, None, :] - alpha[..., None]*predicted
    candidate = (alpha[..., None, None]*memory +
                 beta[..., None, None]*residual[..., None]*key[..., None, :])
    return np.where(admit[:, None, None, None], candidate, memory)


def write_vjp(memory, key, value, alpha, beta, upstream, admitted=True):
    """Analytic VJP of ONE rectangular head. All derivatives are explicit.

    memory[V,K], key[K], value[V], scalar gates; no normalized-key assumption.
    The caller differentiates key normalization/router separately.
    """
    if not admitted:
        return (upstream.copy(), np.zeros_like(key), np.zeros_like(value), 0., 0.)
    old_read = memory @ key
    residual = value - alpha*old_read
    gk = upstream @ key
    dmemory = alpha*(upstream - beta*np.outer(gk, key))
    dkey = beta*(upstream.T @ residual - alpha*memory.T @ gk)
    dvalue = beta*gk
    dalpha = np.sum(upstream*(memory - beta*np.outer(old_read, key)))
    dbeta = residual @ gk
    return dmemory, dkey, dvalue, dalpha, dbeta


@dataclass(frozen=True)
class ReferenceState:
    fast_sum: np.ndarray
    memory: np.ndarray
    clock: np.ndarray
    last_write: np.ndarray
    seen: np.ndarray
    count: int
    aligned: np.ndarray
    owner: np.ndarray
    last_ordinal: np.ndarray


class SourceAlignedModelReference:
    """All learned projections, nonlinearities, clock and memory equations.

    Dense matrices use the source [output,input] storage convention. Readout
    is deterministic regression. No probability/causal-intervention semantics
    or particular training loss is silently attached to it.
    """
    def __init__(self, weights):
        self.w = {k: np.array(v, copy=True) for k, v in weights.items()}
        self.width = self.w["backbone.core.fast_map.weight"].shape[0]//2
        self.state_dim = self.w["historical_content.weight"].shape[0]
        self.dtype = self.w["source_key_rotations"].dtype

    def dense(self, name, x):
        y = x @ self.w[name+".weight"].T
        return y + self.w.get(name+".bias", 0.)

    def core_dense(self, name, x):
        return self.dense("backbone.core."+name, x)

    def initial(self, owners):
        owners = np.asarray(owners)
        if owners.ndim != 1 or owners.dtype != np.int64 or np.any(owners < 0):
            raise ValueError("owner must be nonnegative int64 [B]")
        b = len(owners)
        z = lambda *s: np.zeros(s, dtype=self.dtype)
        return ReferenceState(z(b,self.width),z(b,8,8),z(b),z(b),
                              np.zeros(b,dtype=bool),0,z(b,2,16,8),
                              owners.copy(),np.full(b,-1,dtype=np.int64))

    def keys(self, ordinals):
        phi = ordinal_features(ordinals, self.dtype)
        transformed = np.stack([phi @ r for r in self.w["source_key_rotations"]], -2)
        return normalized(transformed), phi

    def read(self, memory, ordinals):
        keys, phi = self.keys(ordinals)
        head_values = (memory*keys[..., None, :]).sum(axis=-1)
        route = softmax(self.dense("read_router", phi))
        return (head_values*route[..., None]).sum(axis=-2)

    def query(self, state, owners, ordinals):
        ordinals = np.asarray(ordinals)
        if (not np.array_equal(owners,state.owner) or
            ordinals.shape != state.last_ordinal.shape or
            ordinals.dtype != np.int64 or np.any(ordinals < 0) or
            np.any(ordinals > state.last_ordinal)):
            raise ValueError("wrong owner or nonhistorical query")
        return self.dense("historical_content", self.read(state.aligned,ordinals))

    def step(self, s, obs, action, dt, admit, owners, ordinals, query=None):
        b = obs.shape[0]
        if (not np.array_equal(owners,s.owner) or ordinals.dtype != np.int64 or
            ordinals.shape != (b,) or np.any(ordinals <= s.last_ordinal) or
            np.any(ordinals < 0) or admit.dtype != np.bool_ or
            admit.shape != (b,) or dt.shape != (b,) or np.any(dt < 0) or
            not all(np.isfinite(v).all() for v in (obs,action,dt))):
            raise ValueError("invalid source observation/timeline")
        if query is not None and (query.dtype != np.int64 or query.shape != (b,) or
                                  np.any(query < 0) or np.any(query >= ordinals)):
            raise ValueError("query must precede current ordinal")
        now = s.clock + dt
        if not np.isfinite(now).all():
            raise ValueError("clock overflow")
        age = np.maximum(now-s.last_write,0.)
        index = np.full(b,s.count,dtype=self.dtype)
        time = np.stack((np.log1p(index),np.sin(index/16),np.cos(index/16),
                         np.log1p(dt),np.log1p(now),np.log1p(age),
                         s.seen.astype(self.dtype)),axis=-1)
        cat = lambda *xs: np.concatenate(xs,axis=-1)
        e = np.tanh(self.core_dense("encoder",cat(obs,action)) +
                    self.core_dense("time_projection",time))
        k = normalized(self.core_dense("address",e))
        v = normalized(self.core_dense("content",e))
        zr = np.zeros((b,8),dtype=self.dtype)
        zf = np.zeros_like(s.fast_sum)
        split = lambda x: np.split(x,2,axis=-1)
        lerp = lambda x,y,g: x+g*(y-x)
        basis,_ = split(self.core_dense("fast_map",cat(e,zf,zr)))
        basis = np.tanh(basis)
        ctx = s.fast_sum/max(s.count,1)
        p,fg = split(self.core_dense("fast_map",cat(e,ctx,zr)))
        pf = lerp(ctx,np.tanh(p),sigmoid(fg))
        d,g = split(self.core_dense("slow_map",cat(pf,ctx,zr)))
        ps = lerp(ctx,np.tanh(ctx+np.tanh(d)),.25*sigmoid(g))
        alpha,beta = split(sigmoid(self.core_dense("memory_gates",cat(e,pf,ps))))
        alpha = alpha*(1-beta/4)
        r = (s.memory*k[:,None,:]).sum(axis=-1)
        f,fg = split(self.core_dense("fast_map",cat(e,pf,r)))
        f = lerp(pf,np.tanh(f),sigmoid(fg))
        d,g = split(self.core_dense("slow_map",cat(f,ps,r)))
        slow = lerp(ps,np.tanh(ps+np.tanh(d)),.25*sigmoid(g))
        z = cat(f,slow,r,e)
        hidden = self.core_dense("readout.0",z)
        base_delta = np.tanh(self.core_dense("readout.2",hidden*sigmoid(hidden)))
        hidden = self.dense("backbone.residual_ffn.0",z)
        post = sigmoid(self.w["backbone.residual_logit"])*np.tanh(
            self.dense("backbone.residual_ffn.2",hidden*sigmoid(hidden)))
        value = np.tanh(self.dense("source_content",e))
        route = softmax(self.dense("write_router",e))
        a = sigmoid(self.dense("retention",e))
        beta_aligned = sigmoid(self.dense("write_strength",e))*route
        keys,_ = self.keys(ordinals)
        old_ordinal = np.maximum(ordinals-1,0) if query is None else query
        has_prior = ordinals>0 if query is None else np.ones(b,dtype=bool)
        read = np.where(has_prior[:,None],self.read(s.aligned,old_ordinal),0.)
        memory_delta = sigmoid(self.w["predictive_memory_logit"])*np.tanh(
            self.dense("predictive_memory_delta",read))
        scale = self.w["backbone.core.delta_scale"]
        # Preserve the source's order of rounded additions.
        output = obs + scale*base_delta
        output = output + scale*post
        output = output + scale*memory_delta
        native = corrective_write(s.memory[:,None],k[:,None],v,
                                  alpha,beta,admit)[:,0]
        aligned = corrective_write(s.aligned,keys,value,a,beta_aligned,admit)
        after = ReferenceState(s.fast_sum+basis,native,now,
                               np.where(admit,now,s.last_write),s.seen|admit,
                               s.count+1,aligned,s.owner.copy(),ordinals.copy())
        trace = dict(embedding=e,time_features=time,fast=f,slow=slow,old_read=r,
                     aligned_old_read=read,aligned_write_key=keys,
                     aligned_retention=a,aligned_strength=beta_aligned,
                     aligned_write_route=route,aligned_memory=aligned,
                     aligned_before_write=s.aligned,memory=native)
        return output,after,trace

    def forward(self, obs, action, dt, admit, owners, ordinals, query=None):
        s = self.initial(owners)
        outputs, traces = [], []
        for t in range(obs.shape[1]):
            y,s,trace = self.step(s,obs[:,t],action[:,t],dt[:,t],admit[:,t],
                                owners,ordinals[:,t],None if query is None else query[:,t])
            outputs.append(y)
            traces.append(trace)
        return (np.stack(outputs,axis=1),s,
                {k:np.stack([r[k] for r in traces],axis=1) for k in traces[0]})
