"""LATFUSE010 exact-real scalar source-conditional delta-memory reference.

Only an algebraic analogue of original actual tensor recurrent writes;
does not encode trained Torch gradient, FP32 rounding or source correctness.
"""
def source_delta_write(alpha:float,beta:float,key:float,value:float,old:float)->float:
    return alpha*old + beta*(value-alpha*old*key)*key

def source_delta_memory_transfer(alpha:float,beta:float,key:float,old_delta:float)->float:
    return alpha*(1-beta*key*key)*old_delta

def local_event_write_bypass(alpha:float,beta:float,key:float,value:float,old:float)->float:
    return (alpha-1)*old+beta*(value-alpha*old*key)*key

def original_source_state_bytes()->int:
    return 4*(80+80+2*16*8+16*80)+16
