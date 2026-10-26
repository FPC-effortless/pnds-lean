"""LATFUSE002 exact-real-intended local operators, independent of trained Torch.

FP32/64 roundoff, tanh/exponential, AD, probability-law and original-data
source equivalence remain OPEN. Scalar analytical reference tests only.
"""
import math

def multirate_coefficients(width=80, low=.90, high=.99999):
    if width<1 or not 0<low<high<1:
        raise ValueError("invalid retention spectrum")
    if width==1:
        return [high]
    return [math.exp(math.log(low)+(math.log(high)-math.log(low))*i/(width-1))
            for i in range(width)]

def slow_update(previous, candidate, rates):
    if len(previous)!=len(candidate) or len(previous)!=len(rates):
        raise ValueError("slow vector dimension mismatch")
    return [r*x+(1-r)*u for x,u,r in zip(previous,candidate,rates)]

def initialize_memory_gates():
    return dict(retention_logit=math.log(.995/.005),
                write_logit=math.log(.1/.9))

def sigmoid(z):
    return 1/(1+math.exp(-z))

def belief_query(known,belief):
    if len(known)!=len(belief):
        raise ValueError("query dimensions differ")
    return [x+y for x,y in zip(known,belief)]

def contract():
    floats=2*80+16*16+16*80
    return dict(state_float_values=floats,occupancy_flags=16,
                nominal_fp32_bytes=floats*4+16)
