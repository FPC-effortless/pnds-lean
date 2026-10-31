"""Partial SENSE-TOK-006 operator reference, not a full-model formulation.

Selected q and measured losses are evaluator inputs. No argmax/GRU/backward,
finite-precision equivalence, source provenance or capability theorem implied.
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def selected_readout(values: Sequence[Sequence[float]], q: int,
                     permutation: Sequence[int] | None = None) -> tuple[float, ...]:
    if (not values or not values[0] or any(len(v) != len(values[0]) for v in values)
            or any(not math.isfinite(x) for row in values for x in row)
            or type(q) is not int or not 0 <= q < len(values)):
        raise ValueError("finite equal-width code values and legal selected ID required")
    if permutation is not None:
        if (any(type(i) is not int for i in permutation)
                or sorted(permutation) != list(range(len(values)))):
            raise ValueError("permutation must contain every code index exactly once")
        q = permutation[q]
    return tuple(values[q])


def realized_regret(byte_loss: float, token_loss: float, use_byte: bool) -> float:
    if (type(use_byte) is not bool or not math.isfinite(byte_loss)
            or not math.isfinite(token_loss)):
        raise ValueError("finite scalar losses and an explicit selected action required")
    return (byte_loss if use_byte else token_loss) - min(byte_loss, token_loss)


def dense_byte_parameter_count() -> int:
    return 257*32 + 3*118*32 + 3*118*118 + 6*118 + 256*(118+32) + 256
