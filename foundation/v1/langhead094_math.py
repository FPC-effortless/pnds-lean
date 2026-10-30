"""LANGHEAD094 executable scalar and vector diagonal/GRU update reference.

Exact real formulas with Python float arithmetic (NOT verified original 32D
embeddings/128D SSM/GRU trained graph, AD, precision or learned performance).
"""
from __future__ import annotations
import math

def sigmoid(x:float)->float:
    if not math.isfinite(x): raise ValueError("nonfinite gate")
    if x>=0: return 1/(1+math.exp(-x))
    a=math.exp(x);return a/(1+a)
def softplus(x:float)->float:
    if not math.isfinite(x):raise ValueError("nonfinite rate")
    return max(0.,x)+math.log1p(math.exp(-abs(x)))
def diagonal_scalar(previous:float,offer:float,raw_decay:float,raw_input_gate:float)->float:
    if any(not math.isfinite(x) for x in (previous,offer,raw_decay,raw_input_gate)):
        raise ValueError("nonfinite recurrence")
    alpha=math.exp(-softplus(raw_decay)*sigmoid(raw_input_gate))
    return alpha*previous+(1-alpha)*offer
def diagonal_vector(previous,offer,decay,gate):
    if not previous or not(len(previous)==len(offer)==len(decay)==len(gate)):
        raise ValueError("state dimensions mismatch")
    return tuple(diagonal_scalar(*args) for args in zip(previous,offer,decay,gate))
def gru_blend(old:float,candidate:float,update_gate:float)->float:
    if any(not math.isfinite(x) for x in (old,candidate,update_gate)):
        raise ValueError("nonfinite GRU")
    if not 0<=update_gate<=1:raise ValueError("gate outside [0,1]")
    return update_gate*old+(1-update_gate)*candidate
def gru_scalar(old:float,i_r:float,h_r:float,i_z:float,h_z:float,i_n:float,h_n:float)->float:
    if any(not math.isfinite(x) for x in (old,i_r,h_r,i_z,h_z,i_n,h_n)):
        raise ValueError("nonfinite GRU")
    r=sigmoid(i_r+h_r);z=sigmoid(i_z+h_z)
    n=math.tanh(i_n+r*h_n)
    return gru_blend(old,n,z)
