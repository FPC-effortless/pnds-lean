"""LANGMEM078 exact-real independent 8x8 dual-native mathematical reference.

Only independent scalar/rank-one source operators and boundedness checks.
Not original PyTorch full graph/IEEE/admitting missing values or trained model.
"""
from math import exp, log1p, isfinite
K=8
LEGACY_PARAMETERS=499623
DUAL_PARAMETERS=499627
EXTRA_SLOW_FP32_BYTES=4*K*K

def softplus(x):
    if not isfinite(x): raise ValueError("finite raw rate required")
    return max(0.,x)+log1p(exp(-abs(x)))

def rho_from_raw(raw_rate,delta_byte=1):
    if not isfinite(delta_byte) or delta_byte<0:
        raise ValueError("nonnegative source byte distance")
    return exp(-softplus(raw_rate)*delta_byte)

def slow_affine(old,write_key,value,rho,beta,epsilon,admit=True):
    if (len(old)!=K or any(len(row)!=K for row in old)
        or len(write_key)!=K or len(value)!=K or type(admit) is not bool
        or not all(isfinite(x) for x in write_key+value)
        or not all(isfinite(z) for row in old for z in row)
        or not 0<=rho<=1 or not 0<=beta<=.05 or not 0<=epsilon<=.01
        or sum(x*x for x in write_key)>1+1e-7):
        raise ValueError("registered bounded 8x8 source math only")
    if not admit:
        return [list(row) for row in old]
    p=[[rho*((1. if a==b else 0.)-beta*write_key[a]*write_key[b])
        for b in range(K)] for a in range(K)]
    q=[[epsilon*value[a]*write_key[b] for b in range(K)] for a in range(K)]
    return [[sum(old[i][j]*p[j][k] for j in range(K))+q[i][k]
        for k in range(K)] for i in range(K)]

def frobenius(mat):
    return sum(x*x for row in mat for x in row)**.5

def bound_after_steps(initial_norm,rho,epsilon,steps):
    # If ||v||2 <=sqrt(K) and ||wk||2<=1, ||Q||F<=epsilon sqrt(K).
    if not all(isfinite(x) for x in (initial_norm,rho,epsilon)) or initial_norm<0 or not 0<=rho<1 or not 0<=epsilon<=.01 or type(steps)!=int or steps<0:
        raise ValueError("registered bound assumptions")
    return rho**steps*initial_norm+epsilon*(K**.5)*(1-rho**steps)/(1-rho)
