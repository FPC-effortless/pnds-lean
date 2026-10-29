"""LANGMEM080 scoped exact-real single historical write source reference.

The model's native 8x8 source update is affine N_next=N P+Q.
The historical event write contribution C_a = Q_a and C_t = C_(t-1) P_t.
Only source-conditional matrix identities and no-label-leakage timings
are modeled. This does NOT implement source PyTorch FFN, model logits,
source key-learning, byte content reconstruction or full historical replay.
"""
from math import sqrt,isfinite
DIM=8
ANCHORS=(32,64,96,128,192)
DELAYS=(1,8,32,64,128,192)
PAIRS=tuple((a,d) for a in ANCHORS for d in DELAYS if a+d<320)

def matrix_shape(x):
    return len(x)==DIM and all(len(row)==DIM for row in x)

def mul(a,b):
    if not matrix_shape(a) or not matrix_shape(b):raise ValueError("exact original native 8x8 required")
    return [[sum(a[i][k]*b[k][j] for k in range(DIM)) for j in range(DIM)] for i in range(DIM)]

def add(a,b):
    if not matrix_shape(a) or not matrix_shape(b):raise ValueError("same dimensions")
    return [[a[i][j]+b[i][j] for j in range(DIM)] for i in range(DIM)]

def subtract(a,b):
    if not matrix_shape(a) or not matrix_shape(b):raise ValueError("same dimensions")
    return [[a[i][j]-b[i][j] for j in range(DIM)] for i in range(DIM)]

def zero():
    return [[0. for _ in range(DIM)] for _ in range(DIM)]

def step(n,p,q):
    return add(mul(n,p),q)

def tag_step(c,p):
    return mul(c,p)

def norm(mat):
    if not matrix_shape(mat) or not all(isfinite(z) for r in mat for z in r):
        raise ValueError("finite original shape")
    return sqrt(sum(z*z for r in mat for z in r))
