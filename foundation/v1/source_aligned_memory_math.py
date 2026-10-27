"""PAR-MEM-010 independent Python exact-real scalar row/head reference.

This mathematical engineering reference is NOT the PyTorch production graph.
No trained model, probability distribution or authentic generalization claim.
It uses the same known source ordinal modulo periods for lawful lookup.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

PERIODS=(7,13,29,59)


def source_features(ordinal:int)->tuple[float,...]:
    if type(ordinal) is not int or ordinal<0:
        raise ValueError("source ordinal must be nonnegative Python int")
    out=[]
    for period in PERIODS:
        angle=(ordinal % period)*2*math.pi/period
        out.extend((math.cos(angle),math.sin(angle)))
    return tuple(out)


def same_source_address(ordinal:int, rotations):
    phi=source_features(ordinal)
    if len(rotations)!=2 or any(len(m)!=8 or any(len(r)!=8 for r in m) for m in rotations):
        raise ValueError("two 8x8 source-known address maps are required")
    out=[]
    for matrix in rotations:
        v=[sum(phi[i]*matrix[i][j] for i in range(8)) for j in range(8)]
        norm=math.sqrt(sum(z*z for z in v))
        out.append(tuple(z/max(norm,1e-6) for z in v))
    return tuple(out)


def read_row(row,key):
    if len(row)!=8 or len(key)!=8:
        raise ValueError("8-coordinate row/key required")
    return sum(v*k for v,k in zip(row,key))


def gated_row_update(row,key,value,alpha,beta,admit):
    if (len(row)!=8 or len(key)!=8 or
        not 0<=alpha<=1 or not 0<=beta<=1 or type(admit) is not bool):
        raise ValueError("invalid source-restricted gated-delta row")
    if not admit:
        return tuple(row)
    innovation=value-alpha*read_row(row,key)
    return tuple(alpha*r+beta*innovation*k for r,k in zip(row,key))


@dataclass(frozen=True)
class OwnerTimeline:
    owner:int
    last:int=-1

    def accept(self,requested_owner:int,current_ordinal:int):
        if (type(requested_owner) is not int or
            type(current_ordinal) is not int or
            requested_owner!=self.owner or current_ordinal<=self.last):
            raise ValueError("wrong owner or nonmonotonic source event")
        return OwnerTimeline(self.owner,current_ordinal)

    def permit_history(self,requested_owner:int,ordinal:int):
        if (type(requested_owner) is not int or type(ordinal) is not int or
            requested_owner!=self.owner or ordinal<0 or ordinal>self.last):
            raise ValueError("historical source query unauthorized")
        return True
