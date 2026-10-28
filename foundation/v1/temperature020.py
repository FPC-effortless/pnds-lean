"""Independent executable exact-formula reference, FP64 not a whole-model proof."""
import math


def categorical_terms(logits, target, a):
    if a<=0 or not math.isfinite(a):raise ValueError('positive finite inverse temperature')
    shift=max(logits);z=[x-shift for x in logits]
    weights=[math.exp(a*x) for x in z];total=math.fsum(weights)
    p=[w/total for w in weights];mu=math.fsum(w*x for w,x in zip(p,z))
    return (math.log(total)-a*z[target], mu-z[target],
            math.fsum(w*(x-mu)**2 for w,x in zip(p,z)))
