"""Partial exact operator reference; no Torch/model/capability equivalence claim."""
from math import sqrt

def read(matrix,key):
    if len(matrix)!=8 or len(key)!=8 or any(len(row)!=8 for row in matrix):raise ValueError('8x8/8 only')
    return tuple(sum(matrix[i][j]*key[j] for j in range(8)) for i in range(8))

def rank_one(value,key):return tuple(tuple(v*k for k in key) for v in value)

def permute_rows(matrix):return (matrix[-1],)+tuple(matrix[:-1])

def norm_sq(matrix):return sum(x*x for row in matrix for x in row)

def ring_available(next_clock,count,anchor):
    if not 0<=count<=next_clock:raise ValueError('clock/count')
    return next_clock-count<=anchor<next_clock

def capacity(total):
    if total not in (256,4160):raise ValueError('registered total byte budget')
    return total-16

def norm_matched_wrong_event(event,true_value,wrong_value):
    true_sq=sum(x*x for x in true_value);wrong_sq=sum(x*x for x in wrong_value)
    if min(true_sq,wrong_sq)<=0:raise ValueError('nonzero source latent values')
    coefficient=tuple(sum(true_value[i]*event[i][j] for i in range(8))/true_sq for j in range(8))
    rescaled=tuple(x*sqrt(true_sq/wrong_sq) for x in wrong_value)
    return rank_one(rescaled,coefficient)
