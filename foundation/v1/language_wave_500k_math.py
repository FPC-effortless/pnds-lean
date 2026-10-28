"""Independent NumPy mathematical kernels for prospective 500K TAC language.

NOT the 499,290-parameter PyTorch code. These functions are exact algebra
references evaluated in float64, not proof of full Tensor/IEEE/AD refinement.
"""
from __future__ import annotations
import numpy as np

TARGET=500000
MINIMUM=475000
MAXIMUM=525000
CANDIDATE=499290
PERSISTENT_FLOATS=88+8*8+2*16*8+8*2+6*16

def ordered_affine_scan(p:np.ndarray,q:np.ndarray):
    """P[T,D,D], Q[T,R,D], state->state@P+Q, returns inclusive prefixes."""
    if p.ndim!=3 or q.ndim!=3 or p.shape[0]!=q.shape[0] or p.shape[1]!=p.shape[2] or q.shape[-1]!=p.shape[-1]:
        raise ValueError('invalid scan')
    state=np.zeros_like(q[0]);out=[]
    for Pt,Qt in zip(p,q):
        state=state@Pt+Qt;out.append(state.copy())
    return np.stack(out)

def wave_update(old:np.ndarray,theta:float,decay:float,forcing:np.ndarray):
    if old.shape!=(2,) or forcing.shape!=(2,) or not (0<=decay<=1):
        raise ValueError('bounded two-coordinate wave contract')
    c,s=np.cos(theta),np.sin(theta)
    P=decay*np.array([[c,-s],[s,c]])
    return old@P+forcing

def concept_energy(c:np.ndarray,g:np.ndarray,prior:np.ndarray,laplacian:np.ndarray,
                   lam:float=1.,gamma:float=.25,nu:float=.25):
    if lam<=0 or gamma<0 or nu<0:raise ValueError('energy requires positive coefficients')
    return (.5*lam*np.square(c-g).sum()+.5*gamma*np.trace(c.T@laplacian@c)
           +.5*nu*np.square(c-prior).sum())

def concept_equilibrium(g:np.ndarray,prior:np.ndarray,adjacency:np.ndarray,
                        lam:float=1.,gamma:float=.25,nu:float=.25):
    if g.shape!=(6,16) or prior.shape!=(6,16) or adjacency.shape!=(6,6):
        raise ValueError('six concepts x sixteen features required')
    if lam<=0 or gamma<0 or nu<0 or not np.allclose(adjacency,adjacency.T) or (adjacency<0).any():
        raise ValueError('PSD graph contract')
    L=np.diag(adjacency.sum(axis=-1))-adjacency
    H=(lam+nu)*np.eye(6)+gamma*L
    return np.linalg.solve(H,lam*g+nu*prior),L,H

def softmax(logits:np.ndarray):
    m=logits.max(axis=-1,keepdims=True)
    a=np.exp(logits-m)
    return a/a.sum(axis=-1,keepdims=True)
