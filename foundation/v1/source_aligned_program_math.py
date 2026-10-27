"""Executable scalar AD language paired with SourceAlignedDifferentiation.

Real-field totalization (inv(0)=log(0)=sqrt(negative)=0) matches Lean's
mathematical conventions. Regularity excludes singular derivatives. This is
an independent mathematical reference, not a replacement Torch backend.
"""
from dataclasses import dataclass
from fractions import Fraction
import math


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple=()
    payload: object=None

    @staticmethod
    def variable(index):return Expr('input',payload=index)

    @staticmethod
    def constant(value):return Expr('constant',payload=value)

    def value(self,x):
        if self.op=='input':return x[self.payload]
        if self.op=='constant':return self.payload
        a=self.args[0].value(x)
        if self.op=='neg':return -a
        if self.op=='inv':return 0. if a==0 else 1/a
        if self.op=='exp':return math.exp(a)
        if self.op=='log':return 0. if a==0 else math.log(abs(a))
        if self.op=='sqrt':return math.sqrt(max(a,0.))
        if self.op=='sin':return math.sin(a)
        if self.op=='cos':return math.cos(a)
        b=self.args[1].value(x)
        if self.op=='add':return a+b
        if self.op=='mul':return a*b
        if self.op=='max':return max(a,b)
        raise ValueError(self.op)

    def regular(self,x):
        if self.op in ('input','constant'):return True
        if not all(a.regular(x) for a in self.args):return False
        if self.op in ('inv','log','sqrt'):return self.args[0].value(x)!=0
        if self.op=='max':return self.args[0].value(x)!=self.args[1].value(x)
        return True

    def tangent(self,x,dx):
        if self.op=='input':return dx[self.payload]
        if self.op=='constant':return 0.
        a=self.args[0].value(x)
        da=self.args[0].tangent(x,dx)
        if self.op=='neg':return -da
        if self.op=='inv':return 0. if a==0 else -da/(a*a)
        if self.op=='exp':return math.exp(a)*da
        if self.op=='log':return 0. if a==0 else da/a
        if self.op=='sqrt':return 0. if a<=0 else da/(2*math.sqrt(a))
        if self.op=='sin':return math.cos(a)*da
        if self.op=='cos':return -math.sin(a)*da
        b=self.args[1].value(x)
        db=self.args[1].tangent(x,dx)
        if self.op=='add':return da+db
        if self.op=='mul':return da*b+a*db
        if self.op=='max':return db if a<=b else da
        raise ValueError(self.op)


def normalize_program(width=8,epsilon=1e-6):
    v=[Expr.variable(i) for i in range(width)]
    squared=Expr.constant(0.)
    for x in reversed(v):squared=Expr('add',(Expr('mul',(x,x)),squared))
    floor=Expr('max',(squared,Expr.constant(epsilon*epsilon)))
    denominator=Expr('sqrt',(floor,))
    return [Expr('mul',(x,Expr('inv',(denominator,)))) for x in v]


def sum_error_budget(values,ideal=Fraction(0),initial_error=Fraction(0),
                     u=Fraction(1,2**24),eta=Fraction(1,2**150)):
    error=initial_error
    for x in values:
        ideal+=x
        error=(1+u)*error+u*abs(ideal)+eta
    return error
