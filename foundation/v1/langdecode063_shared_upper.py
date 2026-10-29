"""LANGDECODE063 partial executable mathematical reference (not trained TAC source).

Shape agreement: upper input z[280], persistent prior h[48].
Represent the exact order of scalar operations in RecurrenceOnlyUpperBlock.
Function project(z) stands for tanh(input_proj(input_norm(z))) and must be
pure and deterministic. Thus this reference demonstrates one-call reuse, not
the actual trained PyTorch source, IEEE/gradient equivalence or performance.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Callable

Vec = tuple[float, ...]


@dataclass(frozen=True)
class UpperOps:
    project: Callable[[Vec], Vec]
    gate_logits: Callable[[Vec, Vec], Vec]
    output: Callable[[Vec], Vec]
    forget_logits: Callable[[Vec], Vec]
    offer_logits: Callable[[Vec], Vec]
    residual_gain: float


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def _result(z: Vec, prior: Vec, u: Vec, ops: UpperOps) -> tuple[Vec, Vec]:
    if len(z) != 280 or len(prior) != 48 or len(u) != 48:
        raise ValueError("LANGDECODE063 requires z[280], upper[48], u[48]")
    gate_logits = ops.gate_logits(u, prior)
    if len(gate_logits) != 48:
        raise ValueError("gate must have 48 coordinates")
    gated = tuple(math.tanh(a + _sigmoid(g) * p) for a, g, p in zip(u, gate_logits, prior))
    output = ops.output(gated)
    if len(output) != 280:
        raise ValueError("output must have 280 coordinates")
    # Same actual source ordering: source forms residual gain and applies output first.
    gain = _sigmoid(ops.residual_gain)
    refined = tuple(a + gain * v for a, v in zip(z, output))
    forget = ops.forget_logits(u)
    offer = ops.offer_logits(u)
    if len(forget) != 48 or len(offer) != 48:
        raise ValueError("forget/offer must each have 48 coordinates")
    # This performs state transition after the prewrite read; both use prior.
    updated = tuple((.995 * _sigmoid(f)) * p +
                    (1.0 - (.995 * _sigmoid(f))) * math.tanh(v)
                    for f, p, v in zip(forget, prior, offer))
    return refined, updated


def unfused(z: Vec, prior: Vec, ops: UpperOps) -> tuple[Vec, Vec]:
    # Two original pure projections, logically one for read and one for observe.
    u_read = ops.project(z)
    # The reference models the identical read-before-write structure;
    # projected inputs must agree under the purity premise.
    u_write = ops.project(z)
    if u_read != u_write:
        raise ValueError("non-deterministic projection violates reuse premise")
    return _result(z, prior, u_read, ops)


def fused(z: Vec, prior: Vec, ops: UpperOps) -> tuple[Vec, Vec]:
    return _result(z, prior, ops.project(z), ops)
