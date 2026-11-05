"""PNDS standalone R1 mathematical operator tests, no neural training."""
from pathlib import Path
import importlib.util
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('csl125r1_reference',ROOT/'math_reference.py')
ref=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


def test_convex_normalized_memory_bounded():
    rng=np.random.default_rng(113)
    a=rng.dirichlet([.3,1.2,.8],size=(2,100))
    v=rng.uniform(-1,1,size=(2,100,4))
    rates=np.logspace(np.log10(1/8192),np.log10(1/8),12).reshape(3,4)
    read,history,end=ref.serial_weighted_mean_memory(v,a,rates)
    assert np.isfinite(read).all()
    assert np.max(np.abs(read))<=1+1e-12
    assert np.min(history[...,1])>=0
    assert np.max(np.abs(history[...,0])-history[...,1])<=1e-12


def test_oracle_one_hot_slot_persists_without_overwrite():
    a=np.zeros((1,2049,2));a[0,0,0]=1;a[0,1:,1]=1
    v=np.zeros((1,2049,1));v[0,0,0]=.5
    _,_,state=ref.serial_weighted_mean_memory(v,a,np.full((2,1),1/8192))
    ret=state[0,0,0,0]/(state[0,0,0,1]+1e-8)
    assert ret==pytest.approx(.5,abs=1e-3)


def test_bad_oracle_addresses_fail_closed():
    with pytest.raises(ValueError,match='normalized'):
        ref.serial_weighted_mean_memory(np.zeros((1,4,1)),np.ones((1,4,2)),np.full((2,1),.1))
