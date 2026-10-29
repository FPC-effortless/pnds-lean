"""LANGSCALE074 executable partial memory/FFN parameter geometry reference.

Original PyTorch operator equality, IEEE/AD, gates and information survival
are not formalized by these whole-number counts.
"""
from dataclasses import dataclass

TIERS={"h24":(24,480,499625),"h48":(48,455,499623),
       "h96":(96,388,499537),"h128":(128,331,499631)}
BASE_STATE_NATIVE_ASSOCIATIVE_WIDTH=8

@dataclass(frozen=True)
class Geometry:
    upper_state_width:int
    ffn_width:int
    parameters:int
    upper_recurrent_parameters:int
    upper_fp32_bytes_per_owner:int
    native_associative_width:int

def model_count(h:int,w:int)->int:
    if type(h) is not int or type(w) is not int or h<=0 or w<=0:
        raise ValueError("positive exact integer h,w only")
    return 91145+4*h*h+564*h+818*w

def registered(tier:str)->Geometry:
    if tier not in TIERS: raise ValueError("registered four h only")
    h,w,n=TIERS[tier]
    if model_count(h,w)!=n: raise ValueError("model source geometry mismatch")
    return Geometry(h,w,n,841+4*h*h+564*h,4*h,BASE_STATE_NATIVE_ASSOCIATIVE_WIDTH)

def live_gap():
    values=[t[2] for t in TIERS.values()]
    return (max(values)-min(values))/max(values)
