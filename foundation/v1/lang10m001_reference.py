"""Partial exact-operation reference for LANG10M001 upper recurrent block.

No trained capability, source-equivalence or full first-stage math claim.
Source: TAC-OSM PR #658, model source commit 5432befbcd36bb8237ae6929433abb4bd2d9003e.
"""
from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def linear(x, weight, bias):
    return np.matmul(x, np.asarray(weight).T) + np.asarray(bias)


def layernorm(x, scale, bias, eps=1e-5):
    x = np.asarray(x)
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * np.asarray(scale) + np.asarray(bias)


def scalar_step(alpha, offer, previous):
    """Single input-conditioned diagonal recurrence coordinate (Lean scalarStep)."""
    return alpha * previous + (1 - alpha) * offer


def causal_scan(alpha, offers, initial=None):
    """Simple sequential reference for a shape-[B,T,D] gate/offer schedule."""
    alpha = np.asarray(alpha)
    offers = np.asarray(offers)
    if alpha.shape != offers.shape or alpha.ndim != 3:
        raise ValueError("matching nonempty [B,T,D] arrays required")
    b, t, d = alpha.shape
    if t == 0:
        raise ValueError("nonempty chronological sequence required")
    state = np.zeros((b, d), dtype=offers.dtype) if initial is None else np.array(initial)
    prior = []
    after = []
    for i in range(t):
        prior.append(state.copy())
        state = scalar_step(alpha[:, i], offers[:, i], state)
        after.append(state.copy())
    return np.stack(prior, axis=1), np.stack(after, axis=1)


def reference_upper_block(x, p):
    """Partial NumPy formulation of actual recurrent+FFN upper block.

    p is a dictionary of already trained/constructed weights, using the
    source module names (without outer prefix):
      recurrence.{input_norm,input_proj,forget,offer,read_gate,output}
      recurrence.residual_gain
      norm, up, gate, down, ffn_gain

    Input x: shape [B,T,280]. Output: shape [B,T,280]. This is a numerical
    reference, NOT a training implementation. Float precision and gradients
    must be checked against the original PyTorch source separately.
    """
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[2] != 280 or x.shape[1] == 0:
        raise ValueError("upper block expects [B,T,280]")

    def affine(name, z):
        return linear(z, p[name + ".weight"], p[name + ".bias"])

    def norm(name, z):
        return layernorm(z, p[name + ".weight"], p[name + ".bias"])

    u = np.tanh(affine("recurrence.input_proj", norm("recurrence.input_norm", x)))
    alpha = 0.995 * sigmoid(affine("recurrence.forget", u))
    offer = np.tanh(affine("recurrence.offer", u))
    prior, _ = causal_scan(alpha, offer)
    gate = sigmoid(affine("recurrence.read_gate", np.concatenate((u, prior), axis=-1)))
    combined = np.tanh(u + gate * prior)
    z = x + sigmoid(np.asarray(p["recurrence.residual_gain"])) * affine(
        "recurrence.output", combined
    )
    v = norm("norm", z)
    ffn = affine("up", v) * silu(affine("gate", v))
    return z + sigmoid(np.asarray(p["ffn_gain"])) * affine("down", ffn)
