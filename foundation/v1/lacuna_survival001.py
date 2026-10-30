"""Independent exact-equation P0 reference. No full neural proof or efficacy."""
def transport(matrix,key,rho,eta):
 n=len(key)
 return [[rho*(matrix[i][j]-eta*sum(matrix[i][a]*key[a] for a in range(n))*key[j]) for j in range(n)] for i in range(n)]

def source_event(key,value,eta):return [[eta*v*k for k in key] for v in value]

def probability_mixture(p,r,gate):
 if len(p)!=len(r) or not 0<=gate<=1:raise ValueError('matched probability vectors and bounded gate')
 return [(1-gate)*a+gate*b for a,b in zip(p,r)]
