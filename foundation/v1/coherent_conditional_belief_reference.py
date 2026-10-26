"""ME111 independent scalar probability law and coordinate gradient reference.

No neural forward implementation or empirical calibration claim. Arithmetic
is Python binary64; finite-domain checks are explicit, not an IEEE error proof.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from foundation.v1.typed_role_training_reference import (
    _clip_l2, _vector, me110_semantic_masks,
)

EPSILON = 1e-6
LOG4 = math.log(4.0)


def _inputs(vf: float, r: float, vc: float, m: float, raw: float) -> None:
    if not all(math.isfinite(x) for x in (vf, r, vc, m, raw)):
        raise ValueError("inputs must be finite")
    if vf <= 0 or vc <= 0 or m not in (0, 1):
        raise ValueError("positive variances and binary missing indicator required")


def scale(vf: float, r: float, vc: float, m: float, raw: float) -> float:
    """sqrt(epsilon + vf + r² vc m) * exp(log(4) tanh(raw))."""
    _inputs(vf, r, vc, m, raw)
    result = math.sqrt(EPSILON + vf + r * r * vc * m) * math.exp(LOG4 * math.tanh(raw))
    if not math.isfinite(result):
        raise ValueError("scale overflow")
    return result


def scale_vjp(vf: float, r: float, vc: float, m: float, raw: float,
              cotangent: float = 1.0) -> tuple[float, float, float, float]:
    """VJP in (vf, r, vc, raw); missingness is a discrete observed input."""
    sigma = scale(vf, r, vc, m, raw)
    if not math.isfinite(cotangent):
        raise ValueError("cotangent must be finite")
    variance = EPSILON + vf + r * r * vc * m
    shared = cotangent * sigma / (2.0 * variance)
    return (shared, shared * 2.0 * r * vc * m, shared * r * r * m,
            cotangent * sigma * LOG4 * (1.0 - math.tanh(raw) ** 2))


def quantiles(mean: float, left: float, right: float,
              lower_multiplier: float, upper_multiplier: float) -> tuple[float, float, float]:
    """Ordered quantiles for nonnegative tail multipliers and positive scales."""
    if not all(math.isfinite(x) for x in (mean, left, right, lower_multiplier, upper_multiplier)):
        raise ValueError("quantile inputs must be finite")
    if min(left, right) <= 0 or min(lower_multiplier, upper_multiplier) < 0:
        raise ValueError("positive scales and nonnegative multipliers required")
    return mean - lower_multiplier * left, mean, mean + upper_multiplier * right


def semantic_masks(named_shapes: Sequence[tuple[str, Sequence[int]]]) -> tuple[list[list[int]], list[list[int]]]:
    """Return (P,R); R selects mean first six rows of two current families."""
    point = me110_semantic_masks(named_shapes)
    reconstruction = []
    for (name, shape), p in zip(named_shapes, point):
        selected = name in ("current_head.weight", "current_head.bias",
                            "current_residual.output.weight", "current_residual.output.bias")
        reconstruction.append(list(p) if selected else [0] * len(p))
    return point, reconstruction


def route_gradients(point_gradient: Sequence[float], uncertainty_gradient: Sequence[float],
                    reconstruction_mean_gradient: Sequence[float], point_mask: Sequence[int],
                    reconstruction_mask: Sequence[int], *, clip: bool = True) -> list[float]:
    """P*gpoint + .3*R*greconstruction + U*gunc, independently clipped by P/U.

    R must be a subset of P. Reconstruction is added before point-role norm.
    The uncertainty loss includes balanced NLL, interval and reconstruction
    variance; this routine accepts its already differentiated gradient.
    """
    gp = _vector(point_gradient, "point_gradient")
    gu = _vector(uncertainty_gradient, "uncertainty_gradient", len(gp))
    gr = _vector(reconstruction_mean_gradient, "reconstruction_mean_gradient", len(gp))
    p = _vector(point_mask, "point_mask", len(gp))
    r = _vector(reconstruction_mask, "reconstruction_mask", len(gp))
    if any(x not in (0, 1) for x in p + r) or any(ri > pi for pi, ri in zip(p, r)):
        raise ValueError("binary masks with R subset P required")
    point = [pi * a + .3 * ri * b for pi, ri, a, b in zip(p, r, gp, gr)]
    unc = [(1 - pi) * b for pi, b in zip(p, gu)]
    if clip:
        point, unc = _clip_l2(point), _clip_l2(unc)
    return [a + b for a, b in zip(point, unc)]
