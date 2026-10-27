"""Exact chronological upsweep/downsweep falsifiers, including every prefix."""
from fractions import Fraction as F
import unittest
from foundation.v1.source_aligned_memory_math import (
    affine_parameters,compose,apply_affine,
)


def downscan(schedule,incoming):
    def up(xs):
        if len(xs)==1:return xs[0],None,None
        mid=len(xs)//2
        left,right=up(xs[:mid]),up(xs[mid:])
        return compose(left[0],right[0]),left,right
    result=[]
    def down(tree,prefix):
        total,left,right=tree
        if left is None:result.append(compose(prefix,total))
        else:
            down(left,prefix)
            down(right,compose(prefix,left[0]))
    down(up(schedule),incoming)
    return result


class CompletePrefixTests(unittest.TestCase):
    def test_every_inclusive_leaf_padding_and_nonzero_initial_state(self):
        identity=affine_parameters((F(0),)*8,(F(0),)*8,1,0,False)
        initial=tuple(tuple(F(i-j,16) for j in range(8)) for i in range(8))
        for length in (1,2,3,7,8,9,16):
            schedule=[affine_parameters(tuple(F((t+i)%5-2,8) for i in range(8)),
                                         tuple(F(t-i,16) for i in range(8)),
                                         F(7,8),F(1,8),t%3!=1) for t in range(length)]
            padded=1 << (length-1).bit_length()
            pairs=downscan(schedule+[identity]*(padded-length),identity)[:length]
            state=initial
            for event,pair in zip(schedule,pairs):
                state=apply_affine(event,state)
                self.assertEqual(state,apply_affine(pair,initial))

    def test_batch_stride_identity_covers_every_matrix_coordinate(self):
        seen=set()
        for b in range(3):
            for t in range(7):
                for h in range(2):
                    for half in range(2):
                        for r in range(8):
                            for j in range(8):
                                value_row=8*half+r
                                direct=((((b*7+t)*2+h)*16+value_row)*8+j)
                                expanded=(b*7+t)*256+h*128+half*64+r*8+j
                                self.assertEqual(direct,expanded)
                                seen.add(direct)
        self.assertEqual(seen,set(range(3*7*256)))


if __name__=='__main__':unittest.main()
