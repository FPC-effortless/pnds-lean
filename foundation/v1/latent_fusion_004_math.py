"""LATFUSE004 no-model scalar references only; IEEE/AD/full source OPEN."""
def state_bytes(heads):
    if heads not in (2,4): raise ValueError("unknown arm")
    cells=heads*16*(16//heads)
    floats=80+80+cells+16*80
    return dict(heads=heads,key_width=16//heads,matrix=cells,
                fp32_recurrent_values=floats,bytes=floats*4+16)
def hard_edit(selected,old,candidate):
    return candidate if selected else old
def soft_edit(probability,old,candidate):
    if not 0<=probability<=1:raise ValueError("probability outside unit range")
    return (1-probability)*old+probability*candidate
