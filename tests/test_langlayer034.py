"""Source-pinned, constructed no-fit numerical/AD composition; NOT independent AD."""
import os
import sys

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "foundation", "v1"))
sys.path.insert(0, ROOT)
from langlayer034 import serial_residual_layer

TAC_DIR = os.environ.get("TAC_LAYER034_ROOT")
if not TAC_DIR:
    pytest.skip("source-pinned TAC mirror not supplied", allow_module_level=True)
sys.path.insert(0, TAC_DIR)
from layered_transition import CausalResidualTACBlock, LayeredContextTAC034


@pytest.mark.parametrize("width", [24, 48])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_real_source_forward_trace_and_parameter_vjp(width, dtype):
    torch.manual_seed(34000 + width)
    block = CausalResidualTACBlock(280, width).to(dtype).eval()
    z = torch.randn(2, 11, 280, dtype=dtype)
    original, trace = block(z, return_trace=True)
    # Independent parameter objects, no shared source AD graph.
    weights = {n: p.detach().clone().requires_grad_(True)
               for n, p in block.named_parameters()}
    reference, rtrace = serial_residual_layer(weights, z)
    forward_tol = 3e-6 if dtype == torch.float32 else 1e-12
    vjp_tol = 5e-6 if dtype == torch.float32 else 1e-11
    assert torch.max(torch.abs(original-reference)) < forward_tol
    for key in ("state", "prior_state", "concept_posterior", "concept_laplacian"):
        assert torch.max(torch.abs(trace[key]-rtrace[key])) < forward_tol
    phase = torch.arange(original.numel(), dtype=dtype).reshape_as(original) + 1
    scale = phase.mul(.17).sin()
    grad_a = torch.autograd.grad((original * scale).mean(), tuple(block.parameters()))
    grad_b = torch.autograd.grad((reference * scale).mean(), tuple(weights.values()))
    for (name, _), x, y in zip(block.named_parameters(), grad_a, grad_b):
        assert x.isfinite().all() and y.isfinite().all(), name
        assert torch.max(torch.abs(x-y)) < vjp_tol, name


@pytest.mark.parametrize("depth", [2, 3])
def test_actual_full_layered_model_prefix_noninterference(depth):
    torch.manual_seed(34200 + depth)
    model = LayeredContextTAC034(depth=depth).double().eval()
    x = torch.tensor([[5, 17, 21, 6, 4, 25, 3, 9]])
    y = x.clone()
    y[:, 5:] = torch.tensor([[3, 70, 250]])
    with torch.no_grad():
        a = model(x)
        b = model(y)
    assert torch.allclose(a[:, :5], b[:, :5], atol=1e-12, rtol=1e-12)
    assert torch.isfinite(a).all()
