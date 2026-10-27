"""GP-UNC-001 exact scalar reference equations; not TAC-OSM source equivalence."""
from __future__ import annotations
import math


def _finite(*xs):
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in xs):
        raise ValueError("finite real numeric inputs required")


def conditioned_variance(prior: float, cross: float, observed_variance: float) -> float:
    """Var(f*)|y = prior - cross² / observed_variance.

    Mathematical premise: cross² <= prior * observed_variance, observed_variance>0.
    Floating point may incur small discrepancies; not a verified FP error bound.
    """
    _finite(prior, cross, observed_variance)
    if observed_variance <= 0 or prior < 0 or cross*cross > prior*observed_variance + 1e-10:
        raise ValueError("invalid positive covariance premises")
    v = prior - cross*cross/observed_variance
    if not math.isfinite(v) or v < -1e-10:
        raise ValueError("numerically invalid conditional variance")
    return max(0., v)


def precision_increment(old: float, feature: float, noise_variance: float) -> float:
    """Projected scalar GP precision update: old + feature² / noise_variance."""
    _finite(old, feature, noise_variance)
    if noise_variance <= 0 or old <= 0:
        raise ValueError("precision and observation noise must be positive")
    value = old + feature*feature/noise_variance
    if not math.isfinite(value):
        raise ValueError("nonfinite posterior precision")
    return value
