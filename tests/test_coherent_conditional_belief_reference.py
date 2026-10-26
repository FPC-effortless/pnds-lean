"""Independent law, VJP and mask falsifiers for ME111; no training/data."""
import math
import random

import pytest

from foundation.v1.coherent_conditional_belief_reference import (
    quantiles, route_gradients, scale, scale_vjp, semantic_masks,
)


def test_randomized_law_vjp_and_quantile_order():
    rng = random.Random(111)
    for _ in range(150):
        vf, vc = [rng.uniform(.02, 2) for _ in range(2)]
        r, raw = rng.uniform(-2, 2), rng.uniform(-2, 2)
        m = rng.choice([0, 1])
        args = [vf, r, vc, m, raw]
        expected = math.sqrt(1e-6 + vf + r*r*vc*m)*math.exp(math.log(4)*math.tanh(raw))
        assert scale(*args) == expected
        for index, derivative in zip([0, 1, 2, 4], scale_vjp(*args)):
            plus, minus = list(args), list(args)
            plus[index] += 1e-6
            minus[index] -= 1e-6
            numeric = (scale(*plus)-scale(*minus))/2e-6
            assert math.isclose(derivative, numeric, rel_tol=2e-6, abs_tol=2e-8)
        lo, median, hi = quantiles(r, scale(*args), scale(*args), 1.2, 1.5)
        assert lo < median < hi


def test_observed_independence_missing_response_and_no_age_floor():
    assert scale(.2, 0, .3, 0, 0) == scale(.2, 9, 99, 0, 0)
    assert scale_vjp(.2, 9, 99, 0, 0)[1:3] == (0, 0)
    assert scale(.2, 2, .3, 1, 0) > scale(.2, 0, .3, 1, 0)
    assert scale(1e-12, 0, 1, 1, 0) < .002
    assert scale(1e-12, 0, 1, 1, -30) < .0003
    assert scale(.2, -2, .3, 1, 0) == scale(.2, 2, .3, 1, 0)


def test_reconstruction_mask_subset_and_routing():
    shapes = [("current_head.weight", (12, 64)), ("current_head.bias", (12,)),
              ("current_residual.output.weight", (12, 50)),
              ("current_residual.output.bias", (12,)), ("other", (1,))]
    p, r = semantic_masks(shapes)
    assert sum(map(sum, r)) == 696
    assert all(ri <= pi for pt, rt in zip(p, r) for pi, ri in zip(pt, rt))
    assert r[-1] == [0]
    assert route_gradients([2, 2, 2], [3, 3, 3], [5, 5, 5],
                           [1, 1, 0], [1, 0, 0], clip=False) == [3.5, 2, 3]
    with pytest.raises(ValueError):
        route_gradients([1], [1], [1], [0], [1])


@pytest.mark.parametrize("args", [(0, 1, 1, 0, 0), (1, 1, 0, 0, 0),
                                      (1, 1, 1, .5, 0), (float('nan'), 1, 1, 0, 0)])
def test_invalid_domain(args):
    with pytest.raises(ValueError):
        scale(*args)
