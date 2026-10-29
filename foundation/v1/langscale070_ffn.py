"""LANGSCALE070 standard-library executable partial parameter math.

Original TAC fixed first-stage/native/aligned/wave/6 concept memory and
48-dimensional upper block; only the SwiGLU head width varies.
Not a full trained PyTorch/source or finite-precision math proof.
"""
from dataclasses import dataclass

EXPECTED={150:250133,455:499623,1067:1000239}
FIXED=127433
PER_WIDTH=818
UPPER_STATE_WIDTH=48

@dataclass(frozen=True)
class Geometry:
    ffn_width:int
    trainable:int
    upper_width:int
    expansion:int


def counts(ffn_width:int)->Geometry:
    if type(ffn_width) is not int or ffn_width<=0:
        raise ValueError("strict positive integer width required")
    return Geometry(ffn_width, FIXED+PER_WIDTH*ffn_width,UPPER_STATE_WIDTH,
                    PER_WIDTH*ffn_width)


def registered(ffn_width:int)->Geometry:
    if ffn_width not in EXPECTED:
        raise ValueError("only exact three source-registered scales")
    result=counts(ffn_width)
    if result.trainable!=EXPECTED[ffn_width]:
        raise ValueError("source mismatch")
    return result
