"""LANGMEM085 partial independent real-matrix delta law reference.

No trained model, no source corpus, no semantic historical content recovery,
not whole PyTorch IEEE/AD equality. All keys and values must have unit norm.
"""
import math
DIM=8
def inner(a,b):
    return sum(x*y for x,y in zip(a,b))
def matvec(m,k):
    return [inner(row,k) for row in m]
def delta(m,k,v,rho,eta):
    if (len(m)!=8 or any(len(row)!=8 for row in m)
        or len(k)!=8 or len(v)!=8 or not 0<rho<1 or not 0<=eta<=1
        or abs(inner(k,k)-1)>1e-8 or abs(inner(v,v)-1)>1e-8):
        raise ValueError("only registered normalized exact-real source operator")
    prior=matvec(m,k)
    residual=[v[i]-rho*prior[i] for i in range(8)]
    out=[[rho*m[i][j]+eta*residual[i]*k[j] for j in range(8)] for i in range(8)]
    return out,residual
def eta(residual,mode):
    n=math.sqrt(inner(residual,residual))
    if mode=="overwrite":return 1.
    if mode=="partial025":return .25
    if mode=="residual_clip035":return min(1.,.35/(n+1e-9))
    if mode=="selective":return 1. if n<=.75 else .25
    raise ValueError("unregistered original source operator")
