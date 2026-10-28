"""GRAPH007 exact-real reference gates, not PyTorch or IEEE equivalence."""
from math import isfinite

def slow_gates(raw_retention:float,raw_erasure:float,raw_write:float):
    values=(raw_retention,raw_erasure,raw_write)
    if not all(isfinite(v) and 0<=v<=1 for v in values):
        raise ValueError('gate raw values must be finite in [0,1]')
    alpha=.995+.004*raw_retention
    erase=.002*raw_erasure
    write=(1-alpha)*raw_write
    assert .995<=alpha<=.999+1e-15
    assert 0<=erase<=.002
    assert 0<=write<=1-alpha+1e-15
    return alpha,erase,write

def authorized_carry(current_owner:str,stored_owner:str)->bool:
    return bool(current_owner) and current_owner==stored_owner

def old_state_scalar_factor(gates):
    """Upper-bound envelope only, not recoverability or a decoder probability."""
    factor=1.
    for alpha,erase in gates:
        if not (0<=alpha<=1 and 0<=erase<=1):
            raise ValueError('gate out of range')
        factor*=alpha
    return factor

if __name__=='__main__':
    for a in (0.,.25,.5,.75,1.):
        for e in (0.,.5,1.):
            for w in (0.,.5,1.):
                slow_gates(a,e,w)
    assert authorized_carry('chapter-A','chapter-A')
    assert not authorized_carry('chapter-B','chapter-A')
    assert not authorized_carry('','')
    print('GRAPH007 Python source-math gate checks PASS; no full model proof')
