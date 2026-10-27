"""Independent exact-real-intended scalar reference for LATFUSE003 operators.

Local checks, NOT a full trained model/F32/AD/normalization/probability
refinement, no synthetic capability evidence.
"""
def contract():
    matrix=4*16*4
    floats=80+80+matrix+16*80
    return {"head_count":4,"head_key_dim":4,"head_val_dim":16,
            "matrix_cells":matrix,"float_state_count":floats,
            "source_bool_flags":16,"nominal_fp32_bytes":4*floats+16}

def delta(alpha,beta,old,key,value):
    if not 0<=alpha<=1 or not 0<=beta<=1:
        raise ValueError("real gated delta coefficients outside bounds")
    return alpha*old+beta*(value-alpha*old*key)*key

def delta_affine(alpha,beta,old,key,value):
    return alpha*(1-beta*key*key)*old+beta*value*key

def replace(old,selected,candidate):
    if len(old)!=4 or len(selected)!=4 or len(candidate)!=4:
        raise ValueError("all head lists must have length 4")
    return [v if s else x for x,s,v in zip(old,selected,candidate)]
