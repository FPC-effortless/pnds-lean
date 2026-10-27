"""Exact rational falsifiers for scan ordering, rectangular layout and reads."""
from fractions import Fraction as F
import random
import unittest
from foundation.v1.source_aligned_memory_math import (
    affine_parameters,apply_affine,compose,tree_reduce,gated_row_update,
    read_row,source_features,OwnerTimeline,
)


class ExactMemoryVerification(unittest.TestCase):
    def setUp(self):
        self.rng=random.Random(10010)

    def vector(self,n):
        return tuple(F(self.rng.randint(-8,8),16) for _ in range(n))

    def test_corrective_write_equals_rectangular_affine_all_heads_and_rows(self):
        for _ in range(10):
            for admitted in (True,False):
                for head in range(2):
                    key,value=self.vector(8),self.vector(16)
                    rows=tuple(self.vector(8) for _ in range(16))
                    alpha,beta=F(15,16),F(head+1,8)
                    pair=affine_parameters(key,value,alpha,beta,admitted)
                    actual=apply_affine(pair,rows)
                    expected=tuple(gated_row_update(row,key,v,alpha,beta,admitted)
                                   for row,v in zip(rows,value))
                    self.assertEqual(actual,expected)
                    blocks=tuple(apply_affine((pair[0],pair[1][b*8:(b+1)*8]),
                                              rows[b*8:(b+1)*8]) for b in range(2))
                    self.assertEqual(blocks[0]+blocks[1],actual)

    def test_balanced_scan_all_prefixes_nonzero_initial_state_and_odd_lengths(self):
        schedule=[affine_parameters(self.vector(8),self.vector(3),F(3,4),F(1,4),t%3!=1)
                  for t in range(9)]
        initial=tuple(self.vector(8) for _ in range(3))
        state=initial
        for t,pair in enumerate(schedule):
            state=apply_affine(pair,state)
            self.assertEqual(state,apply_affine(tree_reduce(schedule[:t+1]),initial))

    def test_associativity_and_noncommutative_chronology(self):
        a,b,c=[affine_parameters(self.vector(8),self.vector(2),F(7,8),F(3,8),True)
               for _ in range(3)]
        self.assertEqual(compose(compose(a,b),c),compose(a,compose(b,c)))
        self.assertNotEqual(compose(a,b),compose(b,a))

    def test_general_read_interference_equation_including_zero_key(self):
        row,query=self.vector(8),self.vector(8)
        for key in (self.vector(8),(F(0),)*8):
            v,a,b=F(7,8),F(15,16),F(1,4)
            updated=gated_row_update(row,key,v,a,b,True)
            self.assertEqual(read_row(updated,query),a*read_row(row,query)+
                             b*(v-a*read_row(row,key))*read_row(key,query))
            self.assertEqual(v-read_row(updated,key),
                             (1-b*read_row(key,key))*(v-a*read_row(row,key)))

    def test_energy_formula_for_homogeneous_difference(self):
        for _ in range(20):
            d,key=self.vector(8),self.vector(8)
            a,b=F(31,32),F(1,8)
            updated=gated_row_update(d,key,F(0),a,b,True)
            norm=lambda x:sum(v*v for v in x)
            exact=a*a*(norm(d)-b*(2-b*norm(key))*read_row(d,key)**2)
            self.assertEqual(norm(updated),exact)
            self.assertLessEqual(norm(updated),a*a*norm(d))

    def test_unit_key_full_write_is_retrieval_and_erases_previous_value(self):
        key=(F(1),)+(F(0),)*7
        row=self.vector(8)
        first=gated_row_update(row,key,F(3),F(1),F(1),True)
        second=gated_row_update(first,key,F(-2),F(1),F(1),True)
        self.assertEqual(read_row(first,key),3)
        self.assertEqual(read_row(second,key),-2)

    def test_actual_ordinal_period_is_a_content_independent_collision(self):
        self.assertEqual(7*13*29*59,155701)
        for n in (0,1,723,2**50,2**63-1-155701):
            self.assertEqual(source_features(n),source_features(n+155701))

    def test_chronology_allows_a_gap_and_does_not_attest_an_observation(self):
        s=OwnerTimeline(5).accept(5,10).accept(5,20)
        self.assertTrue(s.permit_history(5,15))
        self.assertNotIn(15,{10,20})


if __name__=="__main__":
    unittest.main()
