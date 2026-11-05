"""Standalone PNDS R2 scalar-time checks; no neural model required."""
import numpy as np
import pytest
from math_reference import selective_serial


def test_no_write_holds_nonzero_memory_exact():
    carry=np.array([[[[[0.35,0.7]]]]],dtype=np.float64).reshape(1,1,1,2)
    z=np.zeros((1,3,1,1,2))
    out,last=selective_serial(z,carry,np.array([[0.01]]),np.zeros((1,3,1)))
    assert np.array_equal(last,carry)
    assert np.array_equal(out, np.broadcast_to(carry[:,None], out.shape))


def test_complete_write_is_ordinary_ema():
    r=1/16
    z=np.zeros((1,2,1,1,2));z[0,:,0,0,0]=[1,-1];z[...,1]=1
    out,last=selective_serial(z,np.zeros((1,1,1,2)),np.array([[r]]),np.ones((1,2,1)))
    a=np.exp(-r)
    assert np.allclose(out[0,0,0,0],[1-a,1-a])
    assert np.allclose(out[0,1,0,0],[(1-a)*(a-1),(1-a)*(a+1)])


def test_invalid_gate_must_fail():
    with pytest.raises(ValueError):
        selective_serial(np.zeros((1,1,1,1,2)),np.zeros((1,1,1,2)),np.ones((1,1)),np.full((1,1,1),float('nan')))
