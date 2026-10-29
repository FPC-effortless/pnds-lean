"""LANGSCALE075 source-scope associative shape/count reference, partial.

The original three-bank scan stores 16xk floats each. Only k native
rows carry values; the remaining (16-k) are zero-padded for the batched
three-bank scan. Actual learned source correspondence and IEEE/AD OPEN.
"""
from dataclasses import dataclass
REGISTERED={"m8":(8,499623),"m12":(12,500175),"m16":(16,500663)}
@dataclass(frozen=True)
class Geometry:
    dimension:int
    active_parameters:int
    native_active_matrix_shape:tuple
    aligned_each_shape:tuple
    native_active_fp32_bytes:int
    aligned_two_fp32_bytes:int
    total_three_bank_scan_fp32_bytes:int

def model_count(k:int)->int:
    if type(k) is not int or k not in (8,12,16):
        raise ValueError("registered active dimensions only")
    return 499623+114*(k-8)+(0 if k==8 else 8*k)

def registered(tier:str)->Geometry:
    if tier not in REGISTERED:raise ValueError("unregistered tier")
    k,n=REGISTERED[tier]
    if model_count(k)!=n:raise ValueError("inconsistent original source law")
    return Geometry(k,n,(k,k),(16,k),4*k*k,4*2*16*k,4*3*16*k)

def parity_fraction()->float:
    p=[v[1] for v in REGISTERED.values()]
    return (max(p)-min(p))/max(p)
