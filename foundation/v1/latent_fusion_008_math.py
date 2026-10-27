"""LATFUSE008 math-only reference. NOT end-to-end model proof."""
def counts():
    base=99261
    head=6*16+16+16+1
    matrix=2*16*8
    state=80+80+matrix+16*80
    return dict(associative_params=base+head,nonassociative_params=base+head,
                matrix_elements=matrix,fp32_values=state,
                per_stream_nominal_bytes=state*4+16)

def fast_slow_current_outer(a,b):
    return [[x*y for y in b] for x in a]

def current_only_decode(current,clock,decoder):
    return decoder(current,clock)
