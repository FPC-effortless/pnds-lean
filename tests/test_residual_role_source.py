"""PNDS ME109 equations checked against the actual TAC source, without fits."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
TAC = Path(os.environ.get('TAC_SOURCE_ROOT', '/tmp/tac-me106')).resolve()
sys.path[:0] = [str(TAC), str(TAC / 'src')]
spec = importlib.util.spec_from_file_location('pnds_residual_reference', ROOT / 'foundation/v1/residual_role_reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
spec = importlib.util.spec_from_file_location('tac_residual_source_tests', TAC / 'tests/test_me109_graph_reference.py')
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
source.serial = reference.serial


def test_reference_bytes_and_optional_source_pin():
    assert (ROOT / 'foundation/v1/residual_role_reference.py').read_bytes() == (TAC / 'scripts/me109_graph_reference.py').read_bytes()
    expected = os.environ.get('TAC_SOURCE_COMMIT')
    if expected:
        assert subprocess.check_output(['git', '-C', str(TAC), 'rev-parse', 'HEAD'], text=True).strip() == expected


@pytest.mark.parametrize('dtype', [source.torch.float32, source.torch.float64])
@pytest.mark.parametrize('mask_case', ['mixed', 'last_missing', 'all_missing'])
def test_nonzero_residual_full_source_and_gradients(dtype, mask_case):
    source.test_full_nonzero_residual_graph_parameter_and_input_gradients(dtype, mask_case)


def test_masked_value_invariance():
    source.test_masked_finite_values_do_not_enter_reference_context()
