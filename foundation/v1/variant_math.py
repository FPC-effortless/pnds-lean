"""Executable, finite-size mathematical reference kernels for TAC-OSM/PNDS.

This is *not* the actual trained TAC-OSM Python/PyTorch source. Per-variant
full forward/backward correspondence is unverified until traced separately.
Only the operators under named assumptions are represented here.
Uses stdlib arithmetic; floating-point results are tested, not formally proved.
"""
from __future__ import annotations
import math
from collections.abc import Callable, Sequence


def _finite(*values: float) -> None:
    if not all(math.isfinite(v) for v in values):
        raise ValueError("nonfinite input")


def gated(old: float, target: float, g: float) -> float:
    """h' = (1-g) h + g target, for 0<=g<=1."""
    _finite(old, target, g)
    if not (0.0 <= g <= 1.0):
        raise ValueError("gate must be in [0,1]")
    return (1.0 - g) * old + g * target


def affine_run(a: float, b: float, state: float) -> float:
    """One diagonal/coordinatewise affine update."""
    _finite(a, b, state)
    return a * state + b


def affine_compose(later: tuple[float, float],
                   earlier: tuple[float, float]) -> tuple[float, float]:
    """(a2,b2) o (a1,b1) = (a2*a1,a2*b1+b2), time ordered."""
    a2, b2 = later
    a1, b1 = earlier
    _finite(a1, b1, a2, b2)
    return a2 * a1, a2 * b1 + b2


def quadratic_motion_gate(speed: float, c: float) -> float:
    """gamma(v)=v^2/(v^2+c), c>0. Not a quiet-motion accuracy guarantee."""
    _finite(speed, c)
    if c <= 0:
        raise ValueError("c must be positive")
    return speed * speed / (speed * speed + c)


def source_eor_mass(survival: float,
                    activity_probabilities: Sequence[float]) -> tuple[float, list[float]]:
    """(P(source EOR), activity masses), with normalized conditional activity law."""
    _finite(survival, *activity_probabilities)
    if not 0 <= survival <= 1 or not activity_probabilities:
        raise ValueError("survival in [0,1] and a nonempty activity distribution required")
    if any(p < 0 for p in activity_probabilities) or not math.isclose(
            math.fsum(activity_probabilities), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("activity masses must form a probability distribution")
    return 1.0 - survival, [survival * p for p in activity_probabilities]


def translated_cdf(base_cdf: Callable[[float], float], base_median: float,
                   predicted_median: float, y: float) -> float:
    """F(y)=B(y-m+q); B(q)=1/2 is a PREMISE not checked here."""
    _finite(base_median, predicted_median, y)
    value = float(base_cdf(y - predicted_median + base_median))
    _finite(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("invalid base CDF value")
    return value


def bitemporal_eligible(valid_start: float, valid_end: float,
                        record_start: float, record_end: float,
                        valid_time: float, known_at: float) -> bool:
    """As-known-at admissibility; timestamp provenance is not established here."""
    _finite(valid_start, valid_end, record_start, record_end, valid_time, known_at)
    if not (valid_start <= valid_end and record_start <= record_end):
        raise ValueError("unordered interval")
    return valid_start <= valid_time < valid_end and record_start <= known_at < record_end


def window_readout(weights: Sequence[float], observed: Sequence[float]) -> float:
    """Finite observed window aggregation, not indefinitely persistent memory."""
    if not weights or len(weights) != len(observed):
        raise ValueError("nonempty equal window lengths required")
    _finite(*weights, *observed)
    if any(w < 0 for w in weights) or not math.isclose(
            math.fsum(weights), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("normalized nonnegative weights required")
    return math.fsum(w * x for w, x in zip(weights, observed))


def projected_gated_delta(memory: Sequence[Sequence[float]],
                          key: Sequence[float], value: Sequence[float],
                          alpha: float, beta: float, radius: float) -> list[list[float]]:
    """M' = projection_Frobenius_R(alpha*M + beta*(v-alpha*M*k)*k^T).
    No addressing or long-delay recall guarantee follows from the radius bound.
    """
    _finite(alpha, beta, radius, *key, *value, *(x for row in memory for x in row))
    if not 0 <= alpha <= 1 or not 0 <= beta <= 1 or radius <= 0:
        raise ValueError("invalid retention/write/projection parameters")
    if not memory or not key or len(memory) != len(value) or any(len(row) != len(key) for row in memory):
        raise ValueError("matrix/key/value shape mismatch")
    d = [[alpha * x for x in row] for row in memory]
    predicted = [math.fsum(x * k for x, k in zip(row, key)) for row in d]
    updated = [[x + beta * (v - pred) * k for x, k in zip(row, key)]
               for row, v, pred in zip(d, value, predicted)]
    # hypot handles many finite components more robustly than naive sum of squares.
    norm = math.hypot(*(x for row in updated for x in row))
    scale = min(1.0, radius / norm) if norm > 0 else 1.0
    return [[scale * x for x in row] for row in updated]


def distribution_mass(probabilities: Sequence[float]) -> float:
    """Categorical head total mass only; no continuous multimodal decoder proof."""
    _finite(*probabilities)
    if not probabilities or any(x < 0 for x in probabilities):
        raise ValueError("nonempty nonnegative distribution required")
    return math.fsum(probabilities)


def edge_abs_error(reference: float, edge_value: float) -> float:
    """Observable absolute numerical discrepancy; not a proven edge-runtime bound."""
    _finite(reference, edge_value)
    return abs(reference - edge_value)


def causal_identified(identified_by_valid_scm: bool) -> bool:
    """No observational-to-interventional equality without an identification proof."""
    if not isinstance(identified_by_valid_scm, bool):
        raise TypeError("explicit causal identification status required")
    return identified_by_valid_scm


def independent_real_basis(evidence_tier: str) -> bool:
    """Evidence-tier policy predicate; not a trained-model theorem."""
    return evidence_tier in {"independent_real_test", "measured_operational_use"}


KERNELS = {
    "gated": gated,
    "affine": affine_compose,
    "motion": quadratic_motion_gate,
    "event": source_eor_mass,
    "median": translated_cdf,
    "temporal": bitemporal_eligible,
    "window": window_readout,
    "delta": projected_gated_delta,
    "distribution": distribution_mass,
    "edge": edge_abs_error,
    "causal": causal_identified,
    "qualification": independent_real_basis,
}


def evaluate_kernel(kernel: str, **kwargs):
    """Execute a named partial equation, never infer full-model verification."""
    if kernel not in KERNELS:
        raise ValueError(f"unregistered mathematical kernel: {kernel}")
    return KERNELS[kernel](**kwargs)


def evaluate_variant(variant_id: str, **kwargs):
    """Execute the exact PARTIAL kernel mapped to one pinned TAC-OSM version.

    Never substitute this mathematical reference for the original trained
    graph. Full model equivalence requires a separate original-source trace.
    """
    import json
    from pathlib import Path
    manifest = json.loads((Path(__file__).with_name("variant-math-links.json")).read_text())
    matches = [v for v in manifest["variants"] if v["id"] == variant_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicated TAC-OSM variant: {variant_id}")
    v = matches[0]
    if v["formalization_status"] != "PARTIAL_SHARED_KERNEL_NOT_WHOLE_MODEL":
        raise ValueError("unqualified mathematical-completion label")
    return evaluate_kernel(v["kernel"], **kwargs)
