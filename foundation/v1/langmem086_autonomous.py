"""Independent086 residual-operator reference; full trained graph equivalence OPEN."""
import math

def dot(a,b):return sum(x*y for x,y in zip(a,b))
def read(m,k):return tuple(dot(row,k) for row in m)
def residual_update(m,k,v,rho,eta):
    old=read(m,k)
    return tuple(tuple(rho*m[i][j]+eta*(v[i]-rho*old[i])*k[j] for j in range(8)) for i in range(8))
def clipped_update(m,k,v,rho,gate,cap=.35):
    old=read(m,k);error=tuple(v[i]-rho*old[i] for i in range(8))
    eta=min(1.,cap/(math.sqrt(dot(error,error))+1e-9))*gate
    return residual_update(m,k,v,rho,eta)
def mix_read(f,s,q,w):
    a=read(f,q);b=read(s,q)
    return tuple(w*x+(1-w)*y for x,y in zip(a,b))
def count_parameters(ffn_width=94):return 46189+578*ffn_width
def state_bytes():return 4*(64+64+128+8+48+32+96+8+32)
