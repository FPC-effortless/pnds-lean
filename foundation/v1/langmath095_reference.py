"""Independent exact-equation reference for LANGMATH095 operator diagnostics.

NumPy floating-point execution is not Lean real arithmetic, source PyTorch
parity, a trained model, or evidence for useful persistent intelligence.
"""
from __future__ import annotations

import numpy as np

VARIANT = "LANGMATH095"
TAC_ANCESTOR_COMMIT = "d08fbba33990c503446c2a7d5073219a767f2f62"


def _finite(value, name):
    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.floating) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain finite floating values")
    return arr


def compose(first, second):
    """Chronological row-affine composition: first, then second.

    P is [D,D], Q is [R,D]. The result maps S to S@(P1@P2)+Q1@P2+Q2.
    """
    p1, q1 = (_finite(x, "first pair") for x in first)
    p2, q2 = (_finite(x, "second pair") for x in second)
    if (p1.ndim != 2 or p1.shape[0] != p1.shape[1]
            or p2.shape != p1.shape or q1.ndim != 2
            or q2.shape != q1.shape or q1.shape[1] != p1.shape[0]):
        raise ValueError("expected equal square P and compatible rectangular Q")
    if len({x.dtype for x in (p1, q1, p2, q2)}) != 1:
        raise ValueError("pair dtypes must match")
    return p1 @ p2, q1 @ p2 + q2


def serial(states_p, states_q, initial):
    """Independent chronological oracle, returns every inclusive poststate."""
    ps = _finite(states_p, "P")
    qs = _finite(states_q, "Q")
    s = _finite(initial, "initial").copy()
    if (ps.ndim != 3 or ps.shape[1] != ps.shape[2] or qs.ndim != 3
            or len(ps) != len(qs) or qs.shape[1:] != s.shape
            or s.ndim != 2 or s.shape[1] != ps.shape[1]):
        raise ValueError("incompatible sequence/initial geometry")
    if len({ps.dtype, qs.dtype, s.dtype}) != 1:
        raise ValueError("sequence dtypes must match")
    out = np.empty(qs.shape, dtype=s.dtype)
    for t, (p, q) in enumerate(zip(ps, qs)):
        s = s @ p + q
        out[t] = s
    return out


def bounded_aligned_step(memory, key, value, alpha, erase, write_gate, *, admit=True):
    """M'=alpha M(I-erase kk^T)+(1-alpha)*write_gate*v k^T.

    Sufficient arbitrary-key invariant: ||M||F <= V where ||value||2 <= V,
    ||key||2 <= 1 and all three scalar gates belong to [0,1].
    """
    m = _finite(memory, "memory")
    k = _finite(key, "key")
    v = _finite(value, "value")
    if m.ndim != 2 or k.shape != (m.shape[1],) or v.shape != (m.shape[0],):
        raise ValueError("memory/key/value shape mismatch")
    if len({m.dtype, k.dtype, v.dtype}) != 1:
        raise ValueError("memory/key/value dtypes must match")
    if np.linalg.norm(k) > 1 + 8 * np.finfo(k.dtype).eps:
        raise ValueError("key norm exceeds invariant assumptions")
    if any(not np.isfinite(g) or not 0 <= g <= 1 for g in (alpha, erase, write_gate)):
        raise ValueError("gates must belong to [0,1]")
    if type(admit) is not bool:
        raise ValueError("admission must be a bool")
    if not admit:
        return m.copy()
    p = alpha * (np.eye(len(k), dtype=m.dtype) - erase * np.outer(k, k))
    return m @ p + (1 - alpha) * write_gate * np.outer(v, k)


def concept_solve(evidence, prior, scores):
    """Source equation for six concepts; softmax scores produce graph weights.

    H=1.25I+.25L; C=solve(H,G+.25*prior). Returns C,L,H.
    """
    g = _finite(evidence, "evidence")
    cp = _finite(prior, "prior")
    score = _finite(scores, "scores")
    if g.ndim != 2 or g.shape[0] != 6 or cp.shape != g.shape or score.shape != (6, 6):
        raise ValueError("expected six concept rows, matching prior and 6x6 scores")
    if len({g.dtype, cp.dtype, score.dtype}) != 1:
        raise ValueError("concept dtypes must match")
    sm = np.exp(score - score.max(axis=-1, keepdims=True))
    sm /= sm.sum(axis=-1, keepdims=True)
    adjacency = .5 * (sm + sm.T)
    np.fill_diagonal(adjacency, 0)
    laplacian = np.diag(adjacency.sum(axis=-1)) - adjacency
    h = 1.25 * np.eye(6, dtype=g.dtype) + .25 * laplacian
    return np.linalg.solve(h, g + .25 * cp), laplacian, h


def bounded_log_position(position):
    """Candidate095 replacement log1p(t)/(1+log1p(t)), in [0,1)."""
    t = np.asarray(position)
    numeric_real = (np.issubdtype(t.dtype, np.integer)
                    or np.issubdtype(t.dtype, np.floating))
    if not numeric_real or not np.isfinite(t).all() or (t < 0).any():
        raise ValueError("position must be finite and nonnegative")
    x = np.log1p(t.astype(np.float64))
    return x / (1 + x)


def fixed_key_gain(alpha, erase, write):
    """Diagnostic equilibrium gain, not a variable-key boundedness theorem."""
    if any(not np.isfinite(g) or not 0 <= g <= 1 for g in (alpha, erase, write)):
        raise ValueError("gates must belong to [0,1]")
    denominator = 1 - alpha * (1 - erase)
    if denominator <= 0:
        raise ValueError("no finite equilibrium guarantee")
    return write / denominator
