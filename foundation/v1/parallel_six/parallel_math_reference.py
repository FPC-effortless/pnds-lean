"""Reference-only exact chronological 8x8 affine memory algebra.

Small dependency-free Python numerical oracle for the shared six-family
*memory operator*. NOT a reconstructed PyTorch neural model, Lean proof or
real-world capability experiment. The source PyTorch graph retains its own
gradient/equality tests and explicit unproven full neural formalization.
"""
from __future__ import annotations

N = 8


def matmul(a, b):
    if len(a) != N or len(b) != N or any(len(r) != N for r in a + b):
        raise ValueError("requires two 8x8 matrices")
    return [[sum(a[i][k] * b[k][j] for k in range(N))
             for j in range(N)] for i in range(N)]


def add(a, b):
    return [[a[i][j] + b[i][j] for j in range(N)] for i in range(N)]


def identity():
    return [[float(i == j) for j in range(N)] for i in range(N)]


def zeros():
    return [[0.0 for _ in range(N)] for _ in range(N)]


def apply(matrix, pair):
    p, q = pair
    return add(matmul(matrix, p), q)


def compose(first, second):
    """FIRST pair then SECOND pair; Q1 P2 + Q2, not Q2 P1 + Q1."""
    p, q = first
    p2, q2 = second
    return matmul(p, p2), add(matmul(q, p2), q2)


def schedule(key, value, alpha_raw, beta, *, write=True):
    if not write:
        return identity(), zeros()
    if len(key) != N or len(value) != N:
        raise ValueError("8D key/value required")
    if not 0 <= alpha_raw <= 1 or not 0 <= beta <= 1:
        raise ValueError("raw gate values must lie in [0,1]")
    if sum(k*k for k in key) > 1+1e-12 or sum(v*v for v in value) > 1+1e-12:
        raise ValueError("keys and values require L2 norm <= 1")
    alpha=alpha_raw*(1-beta/4)
    p=[[alpha*((1.0 if i==j else 0.0)-beta*key[i]*key[j])
        for j in range(N)] for i in range(N)]
    q=[[beta*value[i]*key[j] for j in range(N)] for i in range(N)]
    return p,q


def fold(pairs):
    """Chronological operator summary, known finite schedule only."""
    result=identity(),zeros()
    for pair in pairs:
        result=compose(result,pair)
    return result


def step_rank_one(m,key,value,alpha_raw,beta):
    """Equivalent to schedule apply under source-gate assumptions."""
    alpha=alpha_raw*(1-beta/4)
    r=[sum(m[i][j]*key[j] for j in range(N)) for i in range(N)]
    return [[alpha*m[i][j]+beta*(value[i]-alpha*r[i])*key[j]
             for j in range(N)] for i in range(N)]
