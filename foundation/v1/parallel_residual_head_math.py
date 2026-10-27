"""Reference scalar *post-scan* readout only; NOT full model equivalence.

Pure standard-library reference to independent scalar residual mathematics.
No dataset, training, checkpoint, hidden target or simulated ability claim.
"""
from __future__ import annotations

import math


def stable_sigmoid(x: float) -> float:
    if not math.isfinite(x):
        raise ValueError("finite residual gate logit required")
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def predict(base: float, scale: float, logit: float, innovation: float) -> float:
    """Scalar counterpart of base + scale * sigmoid(logit) * tanh(innovation).

    'innovation' is the scalar pre-tanh FFN output, not an observed outcome.
    Real PyTorch/NumPy source correspondence remains a separate obligation.
    """
    if not all(math.isfinite(v) for v in (base, scale, innovation)):
        raise ValueError("finite inputs required")
    if scale < 0.0:
        raise ValueError("TRAIN-defined scale must be nonnegative")
    result = base + (scale * stable_sigmoid(logit)) * math.tanh(innovation)
    if not math.isfinite(result):
        raise ValueError("overflowing post-scan readout")
    return result
