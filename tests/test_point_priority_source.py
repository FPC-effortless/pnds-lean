"""Pinned actual-source checks for the ME106 training computational graph."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
TAC=Path(os.environ.get('TAC_SOURCE_ROOT','/tmp/tac-me106')).resolve()
sys.path[:0]=[str(TAC),str(TAC/'src')]
spec=importlib.util.spec_from_file_location('pnds_priority_reference',ROOT/'foundation/v1/point_priority_reference.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
spec=importlib.util.spec_from_file_location('tac_priority_source_tests',TAC/'tests/test_point_priority_sensor_world.py')
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
source.priority_gradients=reference.priority_gradients

def test_source_pin_when_requested():
    expected=os.environ.get('TAC_SOURCE_COMMIT')
    if not expected:pytest.skip('source pin is required in CI')
    assert subprocess.check_output(['git','-C',str(TAC),'rev-parse','HEAD'],text=True).strip()==expected

@pytest.mark.parametrize('dtype',[source.torch.float32,source.torch.float64])
def test_independent_pnds_source_loss_and_projection_gradients(dtype):
    source.test_independent_full_source_loss_and_projected_gradient_reference(dtype)

def test_view_keeps_values_and_routes_actual_gradients():
    source.test_role_view_preserves_predictions_and_removes_only_direct_median_gradient()

def test_projection_edge_cases_and_finite_precision():
    source.test_projection_conflict_alignment_and_zero_point_boundary()
