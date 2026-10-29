"""LANGDECODE064 partial stdlib math of cap_all positional feature constancy.

This reference models exact ordering semantics and the shape of seven input
features; it is not a trained PyTorch source/IEEE/AD or performance proof.
"""
import math


def phi_position(clock: int) -> tuple[float, ...]:
    if type(clock) is not int or clock < 0:
        raise ValueError("nonnegative integer clock required")
    x=float(clock)
    return (math.log1p(x), math.sin(x/16), math.cos(x/16),
            math.sin(x/64), math.cos(x/64), math.sin(x/256), math.cos(x/256))


def source_capped(clock: int) -> tuple[float, ...]:
    original=phi_position(clock)
    cap=math.log1p(127.0)
    return phi_position(127) if original[0]>cap else original


def cached_capped(clock: int, cached: tuple[float, ...]) -> tuple[float, ...]:
    if len(cached)!=7:
        raise ValueError("seven source-derived coordinates required")
    if type(clock) is not int or clock<0:
        raise ValueError("nonnegative integer clock required")
    return cached if clock>=128 else source_capped(clock)


def build_cached() -> tuple[float, ...]:
    # Strict reference: actual PyTorch caches the full source transform at 128.
    return source_capped(128)
