"""LANGSTREAM076 exact-real partial native write/mask shape math.

Not a substitute for full original PyTorch IEEE/AD source, serialization
or learned memory retention proof. For 8/12/16 only.
"""
VALID=(8,12,16)
def state_floats(k):
    if type(k) is not int or k not in VALID:
        raise ValueError("native source k8/12/16 only")
    return 976+(k*k-64)+33*(k-8)

def native_write(old,p,q,admit):
    k=len(old)
    if type(admit) is not bool or k not in VALID or any(len(row)!=k for row in old):
        raise ValueError("native square memory/admission mismatch")
    if len(p)!=k or len(q)!=k or any(len(row)!=k for row in p+q):
        raise ValueError("native affine shape mismatch")
    if not admit:
        return [list(row) for row in old]
    return [[sum(old[i][m]*p[m][j] for m in range(k))+q[i][j]
             for j in range(k)] for i in range(k)]
