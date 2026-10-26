"""ME111 pinned-source scalar forward/VJP and all-coordinate routing parity."""
from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import pytest
import torch

from foundation.v1.coherent_conditional_belief_reference import (
    route_gradients, scale, scale_vjp, semantic_masks,
)

TAC_ROOT = Path(os.environ.get("ME111_TAC_SOURCE", "/tmp/tac-me110"))
if not (TAC_ROOT / "src/tac_osm/conditional_belief_learning.py").is_file():
    pytest.skip("set ME111_TAC_SOURCE to pinned ME111 source", allow_module_level=True)
sys.path.insert(0, str(TAC_ROOT / "src"))
from tac_osm.conditional_belief_learning import reconstruction_mean_masks, routed_gradients
from tac_osm.typed_role_learning import uncertainty_masks
from tac_osm.variants.conditional_belief_sensor_world import ConditionalBeliefSensorWorld, conditional_scales


@pytest.mark.parametrize("dtype,tolerance", [(torch.float64, 1e-12), (torch.float32, 2e-6)])
def test_scalar_formula_and_all_input_vjps(dtype, tolerance):
    gen = torch.Generator().manual_seed(111)
    for missing in [0., 1.]:
        vf = (.05 + torch.rand((4, 6), dtype=dtype, generator=gen)).requires_grad_()
        vc = (.05 + torch.rand((4, 6), dtype=dtype, generator=gen)).requires_grad_()
        response = torch.randn((4, 6), dtype=dtype, generator=gen, requires_grad=True)
        raw = torch.randn((4, 12), dtype=dtype, generator=gen, requires_grad=True)
        factors = torch.exp(math.log(4.) * raw.tanh())
        left, right, base = conditional_scales(vf, vc, response, torch.full_like(vf, missing), factors)
        for side, offset in [(left, 0), (right, 6)]:
            grads = torch.autograd.grad(side.sum(), (vf, response, vc, raw), retain_graph=True)
            for i in range(4):
                for j in range(6):
                    args = (vf[i,j].item(), response[i,j].item(), vc[i,j].item(), missing, raw[i,j+offset].item())
                    assert math.isclose(side[i,j].item(), scale(*args), rel_tol=tolerance, abs_tol=tolerance)
                    want = scale_vjp(*args)
                    actual = [g[i,j].item() for g in grads[:3]] + [grads[3][i,j+offset].item()]
                    assert all(math.isclose(a,b,rel_tol=tolerance,abs_tol=tolerance) for a,b in zip(actual,want))
                    assert grads[3][i,j+6-offset].item() == 0
        assert torch.all(base > 0)


@pytest.mark.parametrize("dtype,tolerance", [(torch.float64, 1e-11), (torch.float32, 3e-6)])
def test_all_masks_and_routed_gradients_match_source(dtype, tolerance):
    torch.set_num_threads(1)
    model = ConditionalBeliefSensorWorld().to(dtype=dtype)
    named = list(model.named_parameters())
    point, recon = semantic_masks([(name, tuple(p.shape)) for name,p in named])
    source_u, source_r = uncertainty_masks(model), reconstruction_mean_masks(model)
    assert len(named) == 45
    assert sum(p.numel() for _,p in named) == 104663
    assert sum(sum(m) for m in recon) == 696
    assert sum(sum(1-b for b in m) for m in point) == 4410
    gp_flat, gu_flat, gr_flat, p_flat, r_flat = [], [], [], [], []
    point_terms, unc_terms, recon_terms = [], [], []
    for i, ((name, parameter), pm, rm, um, sr) in enumerate(zip(named, point, recon, source_u, source_r)):
        assert um.reshape(-1).tolist() == [not bool(b) for b in pm], name
        assert sr.reshape(-1).tolist() == [bool(b) for b in rm], name
        gp = torch.tensor(pm, dtype=dtype).reshape_as(parameter) * ((i+1)/100.)
        gu = torch.full_like(parameter, (i%7+1)/100.)
        gr = torch.full_like(parameter, (i%5+1)/100.)
        point_terms.append((parameter*gp).sum())
        unc_terms.append((parameter*gu).sum())
        recon_terms.append((parameter*gr).sum())
        gp_flat.extend(gp.reshape(-1).tolist()); gu_flat.extend(gu.reshape(-1).tolist())
        gr_flat.extend(gr.reshape(-1).tolist()); p_flat.extend(pm); r_flat.extend(rm)
    raw_point, stats = routed_gradients(model, sum(point_terms), sum(unc_terms), sum(recon_terms))
    expected = route_gradients(gp_flat,gu_flat,gr_flat,p_flat,r_flat)
    actual = [v for _,p in named for v in p.grad.reshape(-1).tolist()]
    assert all(math.isclose(a,b,rel_tol=tolerance,abs_tol=tolerance) for a,b in zip(actual,expected))
    assert [v for g in raw_point for v in g.reshape(-1).tolist()] == gp_flat
    assert stats['reconstruction_mean_coordinates'] == 696
