"""Independent real-math position feature maps; not PyTorch/IEEE equivalence."""
import math

def at127():
    t=127.
    return [math.log1p(t),math.sin(t/16),math.cos(t/16),math.sin(t/64),math.cos(t/64),math.sin(t/256),math.cos(t/256)]

def transform(features,mode):
    if len(features)!=7: raise ValueError('seven features required')
    cap=math.log(128.)
    if mode=='native': return list(features)
    if mode=='cap_log': return [min(features[0],cap),*features[1:]]
    if mode=='cap_all': return at127() if features[0]>cap else list(features)
    raise ValueError('unknown fixed control')
