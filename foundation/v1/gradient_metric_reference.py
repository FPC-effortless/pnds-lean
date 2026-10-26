"""Exact rational DIAGGRAD001 counterexample, no neural or loss-descent claim."""
from fractions import Fraction


def counterexample():
    p = (Fraction(1), Fraction(1))
    u = (Fraction(2), Fraction(-2))
    rates = (Fraction(1), Fraction(100))
    g = tuple(a+b for a, b in zip(p, u))
    dot = lambda a, b: sum(x*y for x, y in zip(a, b))
    displacement = tuple(-rate*x for rate, x in zip(rates, g))
    return dict(auxiliary_dot=dot(p, u), euclidean_dot=dot(p, g),
                scaled_dot=-dot(p, displacement), update_dot=dot(p, displacement))
