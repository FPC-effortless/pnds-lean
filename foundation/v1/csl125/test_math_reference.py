import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('csl125_math', Path(__file__).with_name('math_reference.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def test_reference_causal_mask():
    assert mod.legal_keys(9,np.arange(12),4).nonzero()[0].tolist()==[6,7,8,9]

def test_reference_ema():
    w=np.array([[[[1.0]],[[2.0]],[[3.0]]]])
    np.testing.assert_allclose(mod.serial_ema(w,np.zeros((1,1,1)),np.array([[0.5]])).ravel(),[0.5,1.25,2.125])
