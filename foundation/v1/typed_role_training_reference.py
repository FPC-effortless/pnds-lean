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


def _clip_l2(values: list[float]) -> list[float]:
    norm = math.sqrt(math.fsum(value * value for value in values))
    if not math.isfinite(norm):
        raise ValueError("gradient norm must be finite")
    scale = min(1.0, 1.0 / (norm + 1e-6))
    return [value * scale for value in values]


def route_gradients(point_gradient: Sequence[float],
                    uncertainty_gradient: Sequence[float],
                    mask: Sequence[int | float]) -> list[float]:
    """Compute P_i*gpoint_i + (1-P_i)*gunc_i for a binary mask.

    The coordinate mask is applied to each raw gradient first, then the point
    and uncertainty vectors are independently scaled by min(1, 1/(norm+1e-6)),
    matching the source.
    """
    point = _vector(point_gradient, "point_gradient")
    unc = _vector(uncertainty_gradient, "uncertainty_gradient", len(point))
    roles = _vector(mask, "mask", len(point))
    if any(role not in (0.0, 1.0) for role in roles):
        raise ValueError("mask entries must be exactly 0 or 1")
    point_role = [role * gp for role, gp in zip(roles, point)]
    unc_role = [(1.0 - role) * gu for role, gu in zip(roles, unc)]
    point_role = _clip_l2(point_role)
    unc_role = _clip_l2(unc_role)
    return [gp + gu for gp, gu in zip(point_role, unc_role)]


def me110_semantic_masks(
    named_shapes: Sequence[tuple[str, Sequence[int]]],
) -> list[list[int]]:
    """Return one row-major scalar-coordinate mask per named parameter.

    This mirrors the pinned source's 45-tensor rule: exact name ``state_noise``
    and all width_head/width_residual tensors are uncertainty; rows 6 through
    11 of the four twelve-row output families are uncertainty; all remaining
    scalar coordinates are point-role. A mask value of one means point-role.
    """
    unc_prefixes = (
        "state_noise.", "width_head.", "width_residual.",
    )
    unc_output_prefixes = (
        "readout.2.", "retrieved_delta.2.", "current_head.",
        "current_residual.output.",
    )
    masks: list[list[int]] = []
    for name, shape_values in named_shapes:
        shape = tuple(int(dim) for dim in shape_values)
        if any(dim < 0 for dim in shape):
            raise ValueError(f"{name} has invalid shape")
        count = math.prod(shape) if shape else 1
        uncertainty = name == "state_noise" or name.startswith(unc_prefixes)
        output_family = any(name.startswith(prefix) for prefix in unc_output_prefixes)
        if output_family:
            if not shape or shape[0] != 12:
                raise ValueError(f"{name} must have twelve output rows")
            row_width = math.prod(shape[1:]) if len(shape) > 1 else 1
            masks.append([0 if index // row_width >= 6 else 1
                          for index in range(count)])
        else:
            masks.append([0 if uncertainty else 1] * count)
    return masks


def flatten_masks(masks: Sequence[Sequence[int]]) -> list[int]:
    """Flatten tensor masks in the source named-parameter order."""
    return [bit for tensor_mask in masks for bit in tensor_mask]
