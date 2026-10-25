"""Source-pinned ME104 numerical checks using the independent PNDS reference."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TAC = Path(os.environ.get("TAC_SOURCE_ROOT", "/tmp/tac-me104")).resolve()
sys.path[:0] = [str(TAC), str(TAC / "src"), str(ROOT / "foundation/v1")]

reference_spec = importlib.util.spec_from_file_location(
    "pnds_median_split_sensor_reference",
    ROOT / "foundation/v1/median_split_sensor_reference.py",
)
reference = importlib.util.module_from_spec(reference_spec)
reference_spec.loader.exec_module(reference)

source_test_spec = importlib.util.spec_from_file_location(
    "tac_me104_source_tests", TAC / "tests/test_median_split_sensor_world.py"
)
source_tests = importlib.util.module_from_spec(source_test_spec)
source_test_spec.loader.exec_module(source_tests)
# Reuse TAC's actual-source tests, replacing only its evaluator with the
# separately stored PNDS serial reference.
source_tests.reference_serial = reference.serial


def test_source_pin_when_requested():
    expected = os.environ.get("TAC_SOURCE_COMMIT")
    if not expected:
        pytest.skip("TAC_SOURCE_COMMIT is unset; run in the pinned workflow")
    actual = subprocess.check_output(
        ["git", "-C", str(TAC), "rev-parse", "HEAD"], text=True
    ).strip()
    assert actual == expected


@pytest.mark.parametrize("dtype", [source_tests.torch.float32, source_tests.torch.float64])
def test_pnds_reference_against_actual_source_forward_state_and_gradients(dtype):
    source_tests.test_independent_reference_actual_source_forward_states_and_gradients(dtype)


def test_actual_source_zero_head_parent_warm_start_and_parameter_budget():
    source_tests.test_parameter_budget_and_zero_head_exact_parent_warm_start()


def test_actual_source_quantile_cdf_and_zero_atom_semantics():
    source_tests.test_quantiles_order_cdf_consistency_and_zero_atom()
    source_tests.test_zero_atom_uses_right_side_when_threshold_is_above_median()
    source_tests.test_probability_and_target_validation()


def test_actual_source_split_nll_edges_and_gradient():
    source_tests.test_split_normal_halves_each_integrate_to_one_half_and_nll_is_finite()


def test_actual_source_chunking_and_median_location_invariance():
    source_tests.test_chunked_parity_and_width_head_does_not_change_sensor_memory()
    source_tests.test_nonzero_width_changes_tails_but_never_the_parent_median_location()
