"""Partial LANGMEM088 exact two-logit preference reference, not ML proof.

No actual Rust source facts, no learned optimization, no historical recall
claim. Real 100521 model read_mix.bias[1] init changes by +1 only.
"""
import math
TRAINABLE=100521
PERSISTENT_FP32_BYTES=1920

def slow_share(fast_logit,slow_logit,extra=0.0):
    a=float(fast_logit);b=float(slow_logit)+float(extra)
    # numerically stable two-way softmax
    d=max(a,b)
    e_fast=math.exp(a-d)
    e_slow=math.exp(b-d)
    return e_slow/(e_fast+e_slow)

def slow_share_derivative(fast_logit,slow_logit,extra=0.):
    p=slow_share(fast_logit,slow_logit,extra)
    return p*(1-p)

def apply_bias(source_bias,delta=1.):
    if len(source_bias)!=2:raise ValueError("original learned mixer has exactly two source coordinates")
    return (float(source_bias[0]),float(source_bias[1])+delta)
