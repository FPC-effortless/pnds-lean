import importlib.util
from pathlib import Path


def test_exact_counterexample_matches_lean_constants():
    path = Path(__file__).resolve().parents[1]/'foundation/v1/gradient_metric_reference.py'
    spec = importlib.util.spec_from_file_location('gradient_metric', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.counterexample() == dict(auxiliary_dot=0, euclidean_dot=2,
                                           scaled_dot=-97, update_dot=97)
