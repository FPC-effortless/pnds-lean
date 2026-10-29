"""LANGMEM077 partial exact-real native retained-difference reference.

This is intentionally an independent finite matrix/real-number reference,
NOT the full PyTorch trained graph. The exact contraction bound assumes
normalized write key, beta in [0,1], alpha in [0,1] and same source P/Q.
"""
import math

def native_p(alpha,beta,w):
    if (not isinstance(w,(tuple,list)) or not w or
        len(w) not in (8,12,16) or
        not all(type(x) in (float,int) and math.isfinite(x) for x in w)
        or not math.isfinite(alpha) or not math.isfinite(beta)
        or not 0<=alpha<=1 or not 0<=beta<=1):
        raise ValueError("bounded native original gate/shape assumptions")
    if sum(x*x for x in w)>1+1e-9:
        raise ValueError("source previous_key not normalized")
    k=len(w)
    return [[alpha*((1. if i==j else 0.)-beta*w[i]*w[j])
             for j in range(k)] for i in range(k)]

def right_product(a,p):
    k=len(a)
    if len(p)!=k or any(len(row)!=k for row in a+p):
        raise ValueError("square native matrix mismatch")
    return [[sum(a[i][u]*p[u][j] for u in range(k))
             for j in range(k)] for i in range(k)]

def step(n,p,q):
    return [[x+y for x,y in zip(rowx,rowy)]
            for rowx,rowy in zip(right_product(n,p),q)]

def difference(a,b):
    return [[x-y for x,y in zip(ra,rb)] for ra,rb in zip(a,b)]

def frobenius(a):
    return math.sqrt(sum(z*z for row in a for z in row))

def source_contraction_bound(delta,p,alpha):
    after=right_product(delta,p)
    return frobenius(after),alpha*frobenius(delta)
