"""Tiny independent exact integer count/interface reference, NOT neural proof."""
def contract():
    matrix=2*16*8
    states=80+80+matrix+16*80
    new=6*16+16+16*1+1
    return dict(matrix_cells=matrix,carried_fp32_values=states,
                source_flags=16,nominal_bytes=states*4+16,
                base_live_parameters=99261,added_query_parameters=new,
                candidate_live_parameters=99261+new)
def label_free_query(read,current_state,historic_known_clock):
    return read(current_state,historic_known_clock)
