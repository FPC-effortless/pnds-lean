"""Exact pinned TAC source against PNDS's independent role tensor equations."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
TAC=Path(os.environ.get('TAC_SOURCE_ROOT','/tmp/tac-me106')).resolve()
sys.path[:0]=[str(TAC),str(TAC/'src')]
spec=importlib.util.spec_from_file_location('pnds_role_reference',ROOT/'foundation/v1/role_specialized_reference.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
spec=importlib.util.spec_from_file_location('tac_role_source_tests',TAC/'tests/test_role_specialized_sensor_world.py')
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
source.serial=reference.serial


def test_source_pin_and_reference_bytes():
    expected=os.environ.get('TAC_SOURCE_COMMIT')
    if not expected:pytest.skip('source pin required in CI')
    assert subprocess.check_output(['git','-C',str(TAC),'rev-parse','HEAD'],text=True).strip()==expected
    assert (ROOT/'foundation/v1/role_specialized_reference.py').read_bytes()==(TAC/'scripts/me107_graph_reference.py').read_bytes()


@pytest.mark.parametrize('dtype',[source.torch.float32,source.torch.float64])
def test_pnds_equations_against_actual_full_forward_and_gradients(dtype):
    source.test_full_serial_graph_and_all_parameter_gradients(dtype)


def test_actual_budget_streaming_and_gradient_roles():
    source.test_live_budget_and_hard_assigned_outputs()
    source.test_no_new_recurrent_memory_and_streaming_equivalence()
    source.test_width_and_median_parameter_roles_have_structural_gradient_zeros()


@pytest.mark.parametrize('dtype',[source.torch.float32,source.torch.float64])
def test_actual_displacement_recheck(dtype):
    source.test_blockwise_actual_displacement_contract(dtype)
