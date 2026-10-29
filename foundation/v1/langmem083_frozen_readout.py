"""LANGMEM083 partial independent frozen auxiliary probe size/source math.

NOT PyTorch source, trained-head accuracy, state/AD formal proof, or
qualification of any historical byte decoder. The real script enforces
these same sizes, historical labels and separate FIT/evaluation phases.
"""
AGES=(8,32,128,256)
BASE_PARAMETERS=499627
INPUT=8+8+4
HIDDEN=32
VOCAB=256
TRAINABLE=INPUT*HIDDEN+HIDDEN+HIDDEN*VOCAB+VOCAB
DEPLOYED=BASE_PARAMETERS+TRAINABLE
EXTRA_FP32_BYTES=TRAINABLE*4

def age_onehot(age):
    if type(age) is not int or age not in AGES:
        raise ValueError("registered past age only")
    return tuple(int(a==age) for a in AGES)

def feature(read,writer,age):
    if len(read)!=8 or len(writer)!=8:
        raise ValueError("source read & writer original 8D")
    v=tuple(read)+tuple(writer)+age_onehot(age)
    if len(v)!=INPUT:
        raise ValueError("unexpected original source width")
    return v

def historical_anchor(clock,age):
    if type(clock) is not int or clock<128 or clock>=640:
        raise ValueError("registered source t128..639")
    if age not in AGES or clock-age<4:
        return None
    return clock-age
