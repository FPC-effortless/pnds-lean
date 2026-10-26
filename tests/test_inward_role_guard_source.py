"""Source-pinned tests for the prospective ME109 no-fit inward guard."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from foundation.v1.inward_role_guard import inward_projection

TAC = Path(os.environ.get("TAC_SOURCE_ROOT", "/tmp/tac-me106")).resolve()
TAC_PIN = "f1cf90213cdc9daee161e99e017c55fdcbfb10cc"
TAC_PATH = "src/tac_osm/inward_role_guard.py"
def _pinned_runtime():
    source = TAC / TAC_PATH
    spec = importlib.util.spec_from_file_location("tac_pinned_inward_role_guard", source)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load pinned runtime source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_source_body_matches_pinned_tac_commit():
    resolved = subprocess.check_output(
        ["git", "-C", str(TAC), "rev-parse", f"{TAC_PIN}^{{commit}}"], text=True
    ).strip()
    assert resolved == TAC_PIN
    pinned_body = subprocess.check_output(
        ["git", "-C", str(TAC), "show", f"{TAC_PIN}:{TAC_PATH}"]
    )
    assert (TAC / TAC_PATH).read_bytes() == pinned_body


def test_reference_matches_nonfallback_runtime_candidate_with_recorded_margin():
    runtime = _pinned_runtime()
    weights = [torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32)]
    gradients = [torch.tensor([0.2, -0.5, 0.75], dtype=torch.float32)]
    raw = [torch.tensor([1.9, -2.4, 1.6], dtype=torch.float32)]

    actual, receipts = runtime.inward_candidates(weights, gradients, raw, ((0,),))

    receipt = receipts[0]
    assert receipt["fallback"] is False
    assert receipt["margin"] > 0
    g = gradients[0].double().tolist()
    p = weights[0].double().tolist()
    r = raw[0].double().tolist()
    projected_delta = inward_projection(g, [ri - pi for pi, ri in zip(p, r)], receipt["margin"])
    expected = [pi + di for pi, di in zip(p, projected_delta)]
    assert actual[0].dtype == torch.float32
    assert actual[0].tolist() == pytest.approx(expected, abs=1e-6, rel=0.0)


def test_zero_gradient_block_uses_runtime_path_outside_positive_norm_reference():
    runtime = _pinned_runtime()
    weights = [torch.tensor([1.0, -2.0], dtype=torch.float32)]
    gradients = [torch.zeros(2, dtype=torch.float32)]
    raw = [torch.tensor([3.0, 4.0], dtype=torch.float32)]

    actual, receipts = runtime.inward_candidates(weights, gradients, raw, ((0,),))

    assert torch.equal(actual[0], raw[0])
    assert receipts[0]["norm_squared"] == 0.0
    assert receipts[0]["fallback"] is False
    assert receipts[0]["certified_upper_dot"] <= 0.0
