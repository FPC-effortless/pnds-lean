"""LATFUSE001 independent exact-real-style Python kernel, NOT trained-source parity.

Python floats are finite precision and are not mathematical reals. Numerical
comparisons here only exercise the scalar operator identities in the proposal.
"""
import math


def affine_apply(a, b, z):
    return a * z + b


def affine_compose(first, second):
    a,b = first
    c,d = second
    return c * a, c * b + d


def decay_impute(value, observed, previous, elapsed, mean, rate):
    if not math.isfinite(elapsed) or elapsed < 0 or not math.isfinite(rate) or rate < 0:
        raise ValueError('invalid elapsed/decay')
    if observed:
        if not math.isfinite(value): raise ValueError('nonfinite observation')
        return value
    if not math.isfinite(previous) or not math.isfinite(mean):
        raise ValueError('nonfinite historical value')
    gamma = math.exp(-rate * elapsed)
    return gamma * previous + (1 - gamma) * mean


def gated_delta(matrix, key, value, retain, write):
    if not 0 <= retain <= 1 or not 0 <= write <= 1:
        raise ValueError('gate outside [0,1]')
    if len(matrix) != len(value) or not matrix or len(matrix[0]) != len(key):
        raise ValueError('matrix dimensions')
    norm = math.sqrt(sum(x*x for x in key))
    k = [x/norm for x in key] if norm > 1e-8 else [0.0] * len(key)
    prior = [[retain * x for x in row] for row in matrix]
    pred = [sum(a*b for a,b in zip(row,k)) for row in prior]
    return [[prior[i][j] + write*(value[i]-pred[i])*k[j] for j in range(len(k))]
            for i in range(len(value))]


def source_slot_write(slots, active, index, value, strength):
    if len(slots) != len(active) or index < 0 or index >= len(slots):
        raise ValueError('unregistered source slot')
    if not 0 <= strength <= 1 or len(value) != len(slots[index]):
        raise ValueError('invalid slot write')
    result = [list(row) for row in slots]
    result[index] = [x + strength*(v-x) for x,v in zip(slots[index],value)]
    mask = list(active)
    if strength > 0:
        mask[index] = True
    return result, mask
