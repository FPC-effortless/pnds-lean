"""PNDS NativeMemory v1.1: executable finite-array source-shaped reference.

Pinned source: TAC-OSM commit 081e534a074001c6a3b2251b3ff31ad20ec4fca8
src/tac_osm/aligned_associative_tac.py:gated_write.

Scope: one 8x8 sample, *not* encoder, native fast/slow, optimizer,
batched PyTorch kernel, floating-point bit identity, trained recall or
whole-network gradient equivalence. Formulas are exact-real mathematical
targets, with float64 implementation checks and explicit rounding limits.
"""
from __future__ import annotations
import math
from collections.abc import Sequence

MATRIX_WIDTH = 8
SOURCE_COMMIT = "081e534a074001c6a3b2251b3ff31ad20ec4fca8"
SOURCE_FILE = "src/tac_osm/aligned_associative_tac.py"


def _finite(items: Sequence[float]) -> None:
    if not all(math.isfinite(x) for x in items):
        raise ValueError("input must be finite")


def _validate(matrix, key, value, alpha, beta, radius):
    if len(matrix) != MATRIX_WIDTH or any(len(row) != MATRIX_WIDTH for row in matrix):
        raise ValueError("source store must be 8x8")
    if len(key) != MATRIX_WIDTH or len(value) != MATRIX_WIDTH:
        raise ValueError("source address/value must each have 8 coordinates")
    _finite([alpha, beta, *key, *value, *(v for row in matrix for v in row)])
    if not (0 <= alpha <= 1 and 0 <= beta <= 1):
        raise ValueError("retention and write strength must be in [0,1]")
    if radius is not None and (not math.isfinite(radius) or radius <= 0):
        raise ValueError("radius must be positive if specified")


def read(matrix: Sequence[Sequence[float]], key: Sequence[float]) -> list[float]:
    """Value-by-key orientation, M @ key, as in actual source."""
    if len(key) != MATRIX_WIDTH or len(matrix) != MATRIX_WIDTH or any(
            len(row) != MATRIX_WIDTH for row in matrix):
        raise ValueError("8x8/read shape")
    return [math.fsum(v * k for v, k in zip(row, key)) for row in matrix]


def write(matrix: Sequence[Sequence[float]], key: Sequence[float],
          value: Sequence[float], alpha: float, beta: float,
          writable: bool = True, radius: float | None = 4.0) -> list[list[float]]:
    """One sample. For writable=False the *exact original* matrix is returned.

    For writable=True: D=alpha*M; e=value-D@key;
      U=D+beta*outer(e,key); M'=U/max(1,||U||_F/radius).
    This matches the *mathematical update ordering* of the pinned source.
    """
    _validate(matrix, key, value, alpha, beta, radius)
    if not isinstance(writable, bool):
        raise TypeError("writable must be Bool")
    if not writable:
        return [list(row) for row in matrix]
    d = [[alpha * v for v in row] for row in matrix]
    pred = read(d, key)
    u = [[v + beta * (target - prediction) * k
          for v, k in zip(row, key)]
         for row, target, prediction in zip(d, value, pred)]
    if radius is None:
        return u
    n = math.hypot(*(v for row in u for v in row))
    scale = max(1.0, n / radius)
    return [[v / scale for v in row] for row in u]


def frobenius(matrix: Sequence[Sequence[float]]) -> float:
    return math.hypot(*(v for row in matrix for v in row))


def query_decomposition(matrix, key, value, alpha, beta, query):
    """Read at q BEFORE projection = alpha*M@q + beta*e*(key·q).
    Caller verifies no clip occurred, e is the decayed-key innovation.
    """
    d = [[alpha * v for v in row] for row in matrix]
    innovation = [v - p for v, p in zip(value, read(d, key))]
    overlap = math.fsum(k * q for k, q in zip(key, query))
    old_query = read(matrix, query)
    return [alpha * oq + beta * e * overlap for oq, e in zip(old_query, innovation)]


def self_read_residual(matrix, key, value, alpha, beta):
    """Exact formula requires key dot key = 1 and unprojected update."""
    norm2 = math.fsum(k*k for k in key)
    if not math.isclose(norm2, 1.0, abs_tol=1e-12, rel_tol=1e-12):
        raise ValueError("self-key guarantee requires ||key||²=1")
    r = read(matrix, key)
    return [(1.0 - beta) * (v - alpha * v_old)
            for v, v_old in zip(value, r)]


def finite_difference_value_sensitivity(matrix, key, value, alpha, beta,
                                        query, value_coordinate, eps=1e-6):
    """Numerical derivative of one queried output w.r.t. its input value.

    Analytic derivative is beta*(key·query) without radial projection.
    If the projection is active, this simple derivative does NOT apply.
    """
    if not 0 <= value_coordinate < MATRIX_WIDTH or eps <= 0:
        raise ValueError("invalid finite-difference setup")
    plus = list(value); minus = list(value)
    plus[value_coordinate] += eps
    minus[value_coordinate] -= eps
    result_plus = read(write(matrix, key, plus, alpha, beta, radius=None), query)
    result_minus = read(write(matrix, key, minus, alpha, beta, radius=None), query)
    return ((result_plus[value_coordinate] - result_minus[value_coordinate])/(2*eps),
            beta * math.fsum(k*q for k, q in zip(key, query)))
