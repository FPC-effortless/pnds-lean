"""Constructed math fixtures only; no trained TAC prediction evidence."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "lang10m001_ref", Path(__file__).with_name("lang10m001_reference.py")
)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def test_scalar_step_and_prior_causality():
    a = np.array([[[0.0], [0.5], [1.0], [0.25]]])
    v = np.array([[[1.0], [2.0], [3.0], [4.0]]])
    prior, post = m.causal_scan(a, v)
    assert np.allclose(prior[:, 0], 0)
    assert np.allclose(post[0, :, 0], [1, 1.5, 1.5, 3.375])
    changed = v.copy()
    changed[:, -1] = 100000
    prior2, post2 = m.causal_scan(a, changed)
    assert np.allclose(post[:, :-1], post2[:, :-1])
    assert np.allclose(prior[:, :], prior2[:, :])


def test_wrong_shapes_fail_closed():
    with pytest.raises(ValueError):
        m.causal_scan(np.zeros((1, 3)), np.zeros((1, 3)))
    with pytest.raises(ValueError):
        m.causal_scan(np.zeros((1, 3, 2)), np.zeros((1, 3, 3)))


def fake_params(seed=3):
    rng = np.random.default_rng(seed)
    d, r, f = 280, 128, 1024
    dims = {
        "recurrence.input_proj": (r, d),
        "recurrence.forget": (r, r),
        "recurrence.offer": (r, r),
        "recurrence.read_gate": (r, 2 * r),
        "recurrence.output": (d, r),
        "up": (f, d),
        "gate": (f, d),
        "down": (d, f),
    }
    p = {}
    for k, (out, inc) in dims.items():
        p[k + ".weight"] = rng.normal(size=(out, inc)) / np.sqrt(inc)
        p[k + ".bias"] = np.zeros(out)
    for k, width in (("recurrence.input_norm", d), ("norm", d)):
        p[k + ".weight"] = np.ones(width)
        p[k + ".bias"] = np.zeros(width)
    p["recurrence.residual_gain"] = np.asarray(-2.0)
    p["ffn_gain"] = np.asarray(-2.0)
    return p


def test_upper_reference_shape_and_no_future_dependence():
    rng = np.random.default_rng(44)
    x = rng.normal(size=(1, 5, 280))
    p = fake_params()
    a = m.reference_upper_block(x, p)
    poisoned = x.copy()
    poisoned[:, -1] *= 100
    b = m.reference_upper_block(poisoned, p)
    assert a.shape == x.shape
    assert np.isfinite(a).all()
    assert np.allclose(a[:, :-1], b[:, :-1], atol=1e-9)
