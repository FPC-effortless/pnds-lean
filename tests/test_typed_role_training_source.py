"""Exact coordinate-mask and routed-gradient parity with pinned TAC source."""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest
import torch

from foundation.v1.typed_role_training_reference import (
    flatten_masks,
    me110_semantic_masks,
    route_gradients,
)


TAC_ROOT = Path(os.environ.get("ME110_TAC_SOURCE", "/tmp/tac-me110"))
if not (TAC_ROOT / "src").is_dir():
    pytest.skip("set ME110_TAC_SOURCE to the pinned TAC-OSM checkout", allow_module_level=True)
sys.path.insert(0, str(TAC_ROOT / "src"))

from tac_osm.typed_role_learning import routed_gradients, uncertainty_masks
from tac_osm.variants.residual_role_sensor_world import ResidualRoleSensorWorld


def test_all_45_source_parameter_masks_match_every_scalar_coordinate():
    model = ResidualRoleSensorWorld()
    named = list(model.named_parameters())
    source_masks = uncertainty_masks(model)
    assert len(named) == len(source_masks) == 45
    assert sum(parameter.numel() for _, parameter in named) == 104663

    reference = me110_semantic_masks(
        [(name, tuple(parameter.shape)) for name, parameter in named]
    )
    assert len(reference) == 45
    for (name, parameter), source_mask, reference_mask in zip(
        named, source_masks, reference
    ):
        assert source_mask.numel() == parameter.numel(), name
        source_uncertainty = source_mask.detach().cpu().reshape(-1).tolist()
        assert source_uncertainty == [role == 0 for role in reference_mask], name
    assert sum(reference_mask.count(0) for reference_mask in reference) == 4410
    assert sum(reference_mask.count(1) for reference_mask in reference) == 100253


def test_routed_gradients_match_source_masking_before_epsilon_clipping():
    model = ResidualRoleSensorWorld()
    named = list(model.named_parameters())
    masks = uncertainty_masks(model)
    point_coefficients = []
    uncertainty_coefficients = []
    point_terms = []
    uncertainty_terms = []
    for index, ((_, parameter), mask) in enumerate(zip(named, masks)):
        point = torch.full_like(parameter, float(index + 1))
        point.masked_fill_(mask, 0.0)
        auxiliary = torch.full_like(parameter, float((index % 7) + 2))
        point_coefficients.append(point)
        uncertainty_coefficients.append(auxiliary)
        point_terms.append((parameter * point).sum())
        uncertainty_terms.append((parameter * auxiliary).sum())
    point_loss = torch.stack(point_terms).sum()
    auxiliary_loss = torch.stack(uncertainty_terms).sum()

    routed_gradients(model, point_loss, auxiliary_loss)

    point_flat = [value for tensor in point_coefficients
                  for value in tensor.detach().cpu().reshape(-1).tolist()]
    auxiliary_flat = [value for tensor in uncertainty_coefficients
                      for value in tensor.detach().cpu().reshape(-1).tolist()]
    mask_flat = flatten_masks([
        [0 if bit else 1 for bit in source_mask.detach().cpu().reshape(-1).tolist()]
        for source_mask in masks
    ])
    expected = route_gradients(point_flat, auxiliary_flat, mask_flat)
    actual = [value for _, parameter in named
              for value in parameter.grad.detach().cpu().reshape(-1).tolist()]
    assert len(actual) == 104663
    assert all(math.isclose(got, want, rel_tol=2e-6, abs_tol=2e-7)
               for got, want in zip(actual, expected))


def test_masked_small_vectors_match_source_clip_role_with_epsilon():
    from tac_osm.typed_role_learning import clip_role

    point = [3.0, 4.0, 0.0]
    auxiliary = [9.0, 0.0, 12.0]
    mask = [1, 1, 0]
    expected = route_gradients(point, auxiliary, mask)
    point_tensor = [torch.tensor([point[0], point[1], 0.0], dtype=torch.float64)]
    unc_tensor = [torch.tensor([0.0, 0.0, auxiliary[2]], dtype=torch.float64)]
    clipped_point, _ = clip_role(point_tensor)
    clipped_unc, _ = clip_role(unc_tensor)
    source = (clipped_point[0] + clipped_unc[0]).tolist()
    assert all(math.isclose(got, want, rel_tol=1e-12, abs_tol=1e-12)
               for got, want in zip(source, expected))
