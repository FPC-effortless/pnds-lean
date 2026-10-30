"""LANGHEAD093 exact reference decoders; not a native TAC state implementation.

Matches the subset of model093.py implementing the tied-output score. This
does not prove source-to-source AD or finitary whole-core semantics.
"""
from __future__ import annotations
from typing import Sequence

def dot(a:Sequence[float],b:Sequence[float])->float:
    if len(a)!=len(b) or not a:raise ValueError("dimension mismatch")
    return sum(x*y for x,y in zip(a,b))

def centered(items:Sequence[float])->list[float]:
    if not items:raise ValueError("empty candidates")
    avg=sum(items)/len(items)
    return [v-avg for v in items]

def score_head(embeddings:Sequence[Sequence[float]],observed:int,gain:float,mode:str)->list[float]:
    if len(embeddings)!=256:raise ValueError("byte vocabulary must be 256")
    dim=len(embeddings[0])
    if dim!=32 or any(len(e)!=dim for e in embeddings):raise ValueError("byte embeddings must be length32")
    if not 0<=observed<256 or not 0<=gain<=0.02:raise ValueError("invalid observed byte or gain")
    if mode not in ("tied_dot","tied_dot_norm","norm_only","shuffled_dot","euclidean"):
        raise ValueError("unknown registered head")
    x=embeddings[observed]
    vals=[]
    for y in range(256):
        idx=(73*y+41)%256 if mode=="shuffled_dot" else y
        e=embeddings[idx]
        dp=dot(x,e);norm=dot(e,e)
        if mode=="tied_dot" or mode=="shuffled_dot":s=2*dp
        elif mode=="norm_only":s=-norm
        elif mode=="tied_dot_norm":s=2*dp-norm
        else:s=-sum((ex-ey)**2 for ex,ey in zip(x,e))
        vals.append(s)
    return [gain*v for v in centered(vals)]
