from fractions import Fraction as F
import math
import struct
import unittest
from foundation.v1.source_aligned_program_math import (
    Expr,normalize_program,sum_error_budget,pullback,primitive_error_bound,
)


class ScalarProgramChecks(unittest.TestCase):
    def test_composite_ad_against_independent_centered_difference(self):
        x,y=Expr.variable(0),Expr.variable(1)
        cases=[Expr('exp',(Expr('mul',(x,y)),)),Expr('log',(x,)),
               Expr('sin',(Expr('add',(x,Expr('cos',(y,)))),)),
               Expr('inv',(Expr('sqrt',(x,)),)),Expr('max',(x,y))]
        point=(.7,1.3);direction=(.2,-.3);h=1e-5
        for e in cases:
            self.assertTrue(e.regular(point))
            plus=tuple(a+h*d for a,d in zip(point,direction))
            minus=tuple(a-h*d for a,d in zip(point,direction))
            finite=(e.value(plus)-e.value(minus))/(2*h)
            self.assertAlmostEqual(e.tangent(point,direction),finite,places=9)

    def test_normalizer_zero_is_regular_and_has_identity_over_epsilon_jacobian(self):
        program=normalize_program()
        point=(0.,)*8
        for output,e in enumerate(program):
            self.assertTrue(e.regular(point))
            for input_index in range(8):
                direction=tuple(float(i==input_index) for i in range(8))
                self.assertEqual(e.tangent(point,direction),1e6*float(output==input_index))

    def test_normalizer_boundary_is_not_classically_regular(self):
        point=(1e-6,)+(0.,)*7
        for e in normalize_program():self.assertFalse(e.regular(point))

    def test_clamp_tie_selects_constant_branch_but_is_not_a_classical_derivative(self):
        e=Expr('max',(Expr.variable(0),Expr.constant(0.)))
        self.assertFalse(e.regular((0.,)))
        self.assertEqual(e.tangent((0.,),(1.,)),0.)
        self.assertEqual((e.value((1e-5,))-e.value((0.,)))/1e-5,1.)
        self.assertEqual((e.value((-1e-5,))-e.value((0.,)))/-1e-5,0.)

    def test_rounded_sum_budget_handles_clock_stagnation_and_cancellation(self):
        round32=lambda x:F.from_float(struct.unpack('f',struct.pack('f',float(x)))[0])
        for values in ([F(2**24),F(1)], [F(2**24),F(1),F(-2**24)],
                       [F(1,2**149)]*16,[F(1,3)]*32):
            # Source operands are decoded finite binary values.
            values=[round32(v) for v in values]
            approximate=F(0)
            for value in values:approximate=round32(approximate+value)
            self.assertLessEqual(abs(approximate-sum(values)),sum_error_budget(values))

    def test_singular_expression_is_explicitly_rejected_by_regularity(self):
        e=Expr('inv',(Expr.variable(0),))
        self.assertFalse(e.regular((0.,)))

    def test_full_vjp_pairing_and_shared_input_uses(self):
        x,y=Expr.variable(0),Expr.variable(1)
        outputs=[Expr('mul',(x,Expr('add',(x,y)))),Expr('sin',(y,))]
        point=(.7,1.2);direction=(.3,-.2);upstream=(.4,.9)
        gradient=pullback(outputs,point,upstream)
        self.assertAlmostEqual(sum(g*d for g,d in zip(gradient,direction)),
                               sum(u*e.tangent(point,direction) for u,e in zip(upstream,outputs)))
        self.assertAlmostEqual(gradient[0],.4*(2*.7+1.2))

    def test_nonlinear_forward_bounds_include_clamp_ties_and_domain_margins(self):
        import random
        rng=random.Random(11010)
        for _ in range(100):
            a,b=rng.uniform(.2,2),rng.uniform(.2,2)
            ea,eb=.001,.002
            x,y=a+rng.uniform(-ea,ea),b+rng.uniform(-eb,eb)
            for op,fn in [('sin',math.sin),('cos',math.cos),('exp',math.exp),
                          ('sqrt',math.sqrt),('log',math.log),('inv',lambda z:1/z)]:
                bound=primitive_error_bound(op,(a,),(ea,),margin=min(a,x))
                self.assertLessEqual(abs(fn(a)-fn(x)),bound+1e-15)
            self.assertLessEqual(abs(a*b-x*y),primitive_error_bound('mul',(a,b),(ea,eb)))
            self.assertLessEqual(abs(max(a,b)-max(x,y)),primitive_error_bound('max',(a,b),(ea,eb)))
        with self.assertRaises(ValueError):primitive_error_bound('inv',(1.,),(.1,))
        with self.assertRaises(ValueError):primitive_error_bound('exp',(1.,),(1.1,))


if __name__=='__main__':unittest.main()
