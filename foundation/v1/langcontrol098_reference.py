"""Independent fixed-coefficient objective and logit-gradient equations.

Inherits the objective semantics of LANGMATH098's NumPy controlled_loss, not
that study's new model. LANGCONTROL098 retains the original 086/094/097 graphs.
This computes no model forward, optimizer update, trained score, or IEEE proof.
"""
from __future__ import annotations

import numpy as np

DOMAINS = ('lexical', 'arc')


def coefficient_array(future, valid, domains, *, prefix_weight):
    """Return exact formula coefficients; require nonempty segments/domains."""
    if type(prefix_weight) not in (int, float) or prefix_weight not in (0, 1):
        raise ValueError('the registered prefix coefficient must be zero or one')
    future, valid, domains = map(np.asarray, (future, valid, domains))
    if (future.dtype != bool or valid.dtype != bool or future.ndim != 2
            or valid.shape != future.shape or not future.size
            or domains.shape != (len(future),)
            or set(domains.tolist()) != set(DOMAINS)):
        raise ValueError('both domains and aligned nonempty Boolean masks required')
    prefix = valid & ~future
    if (np.any(future & ~valid) or not np.all(future.sum(1) > 0)
            or not np.all(prefix.sum(1) > 0)
            or np.any(valid[:, 1:] & ~valid[:, :-1])):
        raise ValueError('nonempty prefix/future and contiguous end padding required')
    coefficient = np.zeros(future.shape, dtype=np.float64)
    for domain in DOMAINS:
        indices = np.flatnonzero(domains == domain)
        for i in indices:
            coefficient[i, future[i]] = .5 / len(indices) / future[i].sum()
            coefficient[i, prefix[i]] = (prefix_weight * .5 / len(indices)
                                         / prefix[i].sum())
    return coefficient


def controlled_cross_entropy(logits, targets, future, valid, domains, *, prefix_weight):
    """Return loss and analytic dL/dlogits, independent of source Torch graph.

    All CE positions are computed in both arms. Active coefficient mass is 1
    for lambda=0 and 2 for lambda=1; it is deliberately not renormalized.
    """
    z, y = np.asarray(logits, dtype=np.float64), np.asarray(targets)
    coefficient = coefficient_array(future, valid, domains, prefix_weight=prefix_weight)
    if (z.ndim != 3 or z.shape[:2] != coefficient.shape or z.shape[-1] < 2
            or y.shape != coefficient.shape or y.dtype.kind not in 'iu'
            or not np.isfinite(z).all() or np.any(y < 0) or np.any(y >= z.shape[-1])):
        raise ValueError('finite [B,T,V] logits and aligned legal integer targets required')
    with np.errstate(over='ignore', invalid='ignore'):
        shifted = z - z.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probabilities = exp / exp.sum(axis=-1, keepdims=True)
        terms = (np.log(exp.sum(axis=-1))
                 - np.take_along_axis(shifted, y[..., None], axis=-1)[..., 0])
    if not np.isfinite(terms).all() or not np.isfinite(probabilities).all():
        raise ValueError('cross entropy exceeded finite reference arithmetic')
    gradient = probabilities
    for index in np.ndindex(y.shape):
        gradient[(*index, y[index])] -= 1.
    gradient *= coefficient[..., None]
    with np.errstate(over='ignore', invalid='ignore'):
        loss = float(np.sum(coefficient * terms))
    if not np.isfinite(loss) or not np.isfinite(gradient).all():
        raise ValueError('objective exceeded finite reference arithmetic')
    return loss, gradient
