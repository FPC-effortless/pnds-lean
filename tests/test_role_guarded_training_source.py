"""Source-pinned moment/proposal/recheck equations; toy arithmetic, no TAC fits."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
TAC=Path(os.environ.get('TAC_SOURCE_ROOT','/tmp/tac-me106-source-check')).resolve()
sys.path[:0]=[str(TAC),str(TAC/'src')]
spec=importlib.util.spec_from_file_location('pnds_role_training_ref',ROOT/'foundation/v1/role_guarded_training_reference.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
spec=importlib.util.spec_from_file_location('tac_optimizer_tests',TAC/'tests/test_role_optimizer.py')
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
source.reference_step=reference.step


def test_exact_source_and_reference_pin():
    expected=os.environ.get('TAC_SOURCE_COMMIT')
    if not expected:pytest.skip('CI requires exact source pin')
    assert subprocess.check_output(['git','-C',str(TAC),'rev-parse','HEAD'],text=True).strip()==expected
    assert (ROOT/'foundation/v1/role_guarded_training_reference.py').read_bytes()==(TAC/'scripts/me108_optimizer_reference.py').read_bytes()


@pytest.mark.parametrize('dtype',[source.torch.float32,source.torch.float64])
def test_reference_actual_committed_weights_and_moments(dtype):
    source.test_independent_equations_match_actual_moments_and_applied_parameters(dtype)
    source.test_actual_weight_changes_respect_each_block_after_rounding_and_keep_free_coordinates(dtype)


def test_unguarded_equivalence_and_failure_before_mutation():
    source.test_zero_point_role_recovers_unopposed_torch_adamw_moments_and_weights()
    source.test_invalid_input_cannot_partially_change_weights_or_moments()
