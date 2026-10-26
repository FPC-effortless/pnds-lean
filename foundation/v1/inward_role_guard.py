"""Executable reference for the ME109 inward role-guard projection.

This is a finite-precision Python equation reference, not the trained
TAC-OSM implementation and not a proof of a bound on FP32 rounding error.
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def inward_projection(gradient: Sequence[float], displacement: Sequence[float],
                      margin: float) -> list[float]:
    """Return d' = d - max(0, (g·d + margin)/(g·g)) g.

    Requires equal nonempty finite vectors and strictly positive squared norm.
    The output satisfies the half-space constraint over exact reals; ordinary
    Python float arithmetic can leave a rounding residual.
    """
    if len(gradient) != len(displacement) or not gradient:
        raise ValueError("gradient and displacement must have equal nonzero length")
    values = [*gradient, *displacement, margin]
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("inputs must be finite")
    norm_sq = math.fsum(float(g) * float(g) for g in gradient)
    if not math.isfinite(norm_sq) or norm_sq <= 0.0:
        raise ValueError("gradient must have finite positive squared norm")
    dot = math.fsum(float(g) * float(d) for g, d in zip(gradient, displacement))
    coefficient = max(0.0, (dot + float(margin)) / norm_sq)
    result = [float(d) - coefficient * float(g)
              for g, d in zip(gradient, displacement)]
    if not all(math.isfinite(value) for value in result):
        raise ValueError("projection produced a nonfinite value")
    return result


def rounding_certificate(ideal_dot: float, actual_dot: float, margin: float) -> bool:
    """Check the scalar premise |actual-ideal|≤margin and ideal≤-margin.

    True certifies actual_dot≤0 by scalar arithmetic. The caller must supply
    a justified error bound; this function does not establish one.
    """
    if not all(math.isfinite(float(x)) for x in (ideal_dot, actual_dot, margin)):
        raise ValueError("certificate inputs must be finite")
    return ideal_dot <= -margin and abs(actual_dot - ideal_dot) <= margin and actual_dot <= 0.0
