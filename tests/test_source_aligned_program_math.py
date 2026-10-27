from fractions import Fraction as F
import math
import struct
import unittest
from foundation.v1.source_aligned_program_math import Expr,normalize_program,sum_error_budget


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


if __name__=='__main__':unittest.main()
