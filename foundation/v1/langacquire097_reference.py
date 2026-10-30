"""Independent objective/chunk/gradient mathematics for LANGACQUIRE097.

The byte models' full transitions are source-pinned in LANGACQUIRE097_GRAPH.md.
This NumPy module checks loss semantics and a nontrivial scalar sensitivity
example. It is NOT a reconstruction, fit, or certification of the full TAC,
GRU, SSM, or Transformer, and does not establish a capability advantage.
"""
from __future__ import annotations

import numpy as np


def masked_cross_entropy(logits, targets, mask):
    """Return stable mean active-target CE and its exact-real logit derivative.

    Inactive positions remain in the chronological recurrent input; they have
    zero *direct* loss derivative. Their state may receive a nonzero derivative
    from later active targets. Empty loss masks are invalid rather than PASS.
    """
    z = np.asarray(logits, dtype=np.float64)
    y, active = np.asarray(targets), np.asarray(mask)
    if (z.ndim != 2 or not z.shape[0] or z.shape[1] < 2
            or not np.isfinite(z).all() or y.shape != (len(z),)
            or y.dtype.kind not in "iu" or np.any(y < 0)
            or np.any(y >= z.shape[1]) or active.shape != (len(z),)
            or active.dtype.kind != "b" or not active.any()):
        raise ValueError("finite [T,V] logits, legal integer targets and nonempty Boolean mask required")
    shifted = z - z.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    terms = np.log(exp.sum(axis=1)) - shifted[np.arange(len(z)), y]
    gradient = probabilities.copy()
    gradient[np.arange(len(z)), y] -= 1.0
    gradient *= active[:, None] / int(active.sum())
    return float(terms[active].mean()), gradient


def affine_run(matrix, input_map, initial, observations):
    """Chronological h'=Ah+Bx reference, without a parallel-recurrence claim."""
    a, b = np.asarray(matrix, dtype=float), np.asarray(input_map, dtype=float)
    state = np.asarray(initial, dtype=float).copy()
    x = np.asarray(observations, dtype=float)
    if (a.ndim != 2 or a.shape[0] != a.shape[1] or b.ndim != 2
            or b.shape[0] != len(a) or state.shape != (len(a),)
            or x.ndim != 2 or x.shape[1] != b.shape[1]
            or not all(np.isfinite(value).all() for value in (a, b, state, x))):
        raise ValueError("finite affine-state dimensions required")
    for observed in x:
        state = a @ state + b @ observed
    return state


def scalar_sequence(theta, observations, initial=0.0, initial_derivative=0.0):
    """h_t=theta*h_(t-1)+x_t, retaining BOTH parameter derivative paths."""
    theta = float(theta)
    x = np.asarray(observations, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(theta):
        raise ValueError("finite scalar recurrence required")
    h, derivative = float(initial), float(initial_derivative)
    states, derivatives = [], []
    for observed in x:
        derivative = h + theta * derivative
        h = theta * h + observed
        states.append(h)
        derivatives.append(derivative)
    return np.asarray(states), np.asarray(derivatives)


def scalar_query_loss(theta, acquisition, query, targets):
    """Independent two-class fixture: query-only loss differentiated through A.

    The query readout logits are [h,-h]. Both acquisition and query use the same
    parameter theta, so direct query-parameter effects and carried acquisition
    sensitivity are both present. No optimizer or original-data scoring occurs.
    """
    acquisition = np.asarray(acquisition, dtype=float)
    query = np.asarray(query, dtype=float)
    if not len(acquisition) or not len(query):
        raise ValueError("nonempty acquisition and query required")
    a, da = scalar_sequence(theta, acquisition)
    q, dq = scalar_sequence(theta, query, a[-1], da[-1])
    loss, logit_gradient = masked_cross_entropy(
        np.stack((q, -q), axis=1), targets, np.ones(len(q), dtype=bool))
    derivative = np.sum((logit_gradient[:, 0] - logit_gradient[:, 1]) * dq)
    return loss, float(derivative), {
        "acquired_state": float(a[-1]),
        "acquired_parameter_derivative": float(da[-1]),
        "query_states": q,
        "query_parameter_derivatives": dq,
    }
