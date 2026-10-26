"""Independent executable reference for ME110 coordinate role routing.

The reference applies a binary point-role mask and its complement. Optional
L2 clipping acts separately on each role gradient before routing. It is not a
PyTorch source reconstruction or evidence about full-model AD, IEEE error, or
loss descent.
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def _vector(values: Sequence[float], name: str, n: int | None = None) -> list[float]:
    result = [float(value) for value in values]
    if not result or (n is not None and len(result) != n):
        raise ValueError(f"{name} must have matching nonzero length")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _clip_l2(values: list[float], limit: float = 1.0) -> list[float]:
    norm = math.sqrt(math.fsum(value * value for value in values))
    if not math.isfinite(norm):
        raise ValueError("gradient norm must be finite")
    if norm <= limit or norm == 0.0:
        return values
    scale = limit / norm
    return [value * scale for value in values]


def route_gradients(point_gradient: Sequence[float],
                    uncertainty_gradient: Sequence[float],
                    mask: Sequence[int | float], *,
                    per_role_clip: bool = False) -> list[float]:
    """Compute P_i*gpoint_i + (1-P_i)*gunc_i for a binary mask.

    If ``per_role_clip`` is true, each complete role gradient is independently
    clipped to L2 norm 1 before its coordinates are routed.
    """
    point = _vector(point_gradient, "point_gradient")
    unc = _vector(uncertainty_gradient, "uncertainty_gradient", len(point))
    roles = _vector(mask, "mask", len(point))
    if any(role not in (0.0, 1.0) for role in roles):
        raise ValueError("mask entries must be exactly 0 or 1")
    if per_role_clip:
        point = _clip_l2(point)
        unc = _clip_l2(unc)
    return [role * gp + (1.0 - role) * gu
            for role, gp, gu in zip(roles, point, unc)]


def me110_semantic_mask(parameter_names: Sequence[str]) -> list[int]:
    """Map coordinate labels to roles (labels must identify scalar coordinates).

    Uncertainty coordinates: state_noise; width_head and width_residual;
    output rows 6 in readout.2, retrieved_delta.2, current_head, and
    current_residual.output. Every other coordinate is point-role.
    """
    unc_prefixes = (
        "state_noise.", "width_head.", "width_residual.",
    )
    unc_output_prefixes = (
        "readout.2.", "retrieved_delta.2.", "current_head.",
        "current_residual.output.",
    )
    mask: list[int] = []
    for name in parameter_names:
        is_unc = name.startswith(unc_prefixes)
        if not is_unc:
            for prefix in unc_output_prefixes:
                suffix = name[len(prefix):] if name.startswith(prefix) else ""
                # Accept dotted scalar coordinate labels (e.g. weight.6.2)
                # and bracket labels (e.g. weight[6,2]); require explicit row 6.
                if suffix.startswith("6.") or ".6." in suffix or "[6," in suffix or suffix.endswith("[6]"):
                    is_unc = True
                    break
        mask.append(0 if is_unc else 1)
    return mask
