"""Independent source-to-key and native-state formulation, not whole TAC/IEEE."""
import math


def contextual_keys(tokens,embedding,lag_weights,key_weight,key_bias):
    lexical=[[math.tanh(x) for x in embedding[token]] for token in tokens]
    keys=[]
    for t,e in enumerate(lexical):
        local=[math.tanh(e[c]+math.fsum(math.tanh(lag_weights[l][c])*lexical[t-l-1][c]
                  for l in range(3) if l<t)) for c in range(len(e))]
        raw=[math.fsum(w*x for w,x in zip(row,local))+bias for row,bias in zip(key_weight,key_bias)]
        scale=max(1e-6,math.sqrt(math.fsum(x*x for x in raw)))
        keys.append([x/scale for x in raw])
    return keys


def native_step(memory,key,value,alpha,beta,admit):
    if not admit:return [list(row) for row in memory]
    result=[]
    for row,v in zip(memory,value):
        retained=[alpha*x for x in row]
        residual=v-math.fsum(x*y for x,y in zip(retained,key))
        result.append([x+beta*residual*k for x,k in zip(retained,key)])
    return result


def oblique_erase_counterexample():
    k=[1/math.sqrt(2)]*2;b=[.9,.1];x=[1.,-1.];alpha=.99
    erased=[alpha*(x[i]-k[i]*math.fsum(b[j]*k[j]*x[j] for j in range(2))) for i in range(2)]
    return math.fsum(v*v for v in erased),math.fsum(v*v for v in x)
