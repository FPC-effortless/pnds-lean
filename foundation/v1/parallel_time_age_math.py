"""Independent scalar chronological feature reference for PAR-TIME-001.

Only origin-relative timestamps + most recent previous admitted write.
No per-event factual age claims, no neural training or model predictions.
"""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ClockState:
    count: int = 0
    elapsed: float = 0.0
    last_write: float = 0.0
    seen: bool = False


def advance(state: ClockState, dt: float, writable: bool):
    if (not isinstance(writable,bool) or not isinstance(dt,(int,float))
        or not math.isfinite(dt) or dt<0):
        raise ValueError("source duration must be finite, nonnegative, write boolean")
    if state.count<0 or not (math.isfinite(state.elapsed) and math.isfinite(state.last_write)):
        raise ValueError("invalid previous clock state")
    now=state.elapsed+dt
    if not math.isfinite(now):
        raise ValueError("clock overflow")
    age=now-state.last_write
    if age<0:
        raise ValueError("invalid nonchronological past write")
    i=state.count
    features=(
        math.log1p(i),math.sin(i/16.0),math.cos(i/16.0),
        math.log1p(dt),math.log1p(now),math.log1p(age),
        float(state.seen)
    )
    next_state=ClockState(
        i+1, now, now if writable else state.last_write,
        state.seen or writable
    )
    return features,next_state
