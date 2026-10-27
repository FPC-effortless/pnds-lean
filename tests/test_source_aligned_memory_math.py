import unittest
from foundation.v1.source_aligned_memory_math import (
    source_features,same_source_address,read_row,gated_row_update,OwnerTimeline
)


def identity():
    return tuple(tuple(float(i==j) for j in range(8)) for i in range(8))


class SourceAlignedNoFitMathTests(unittest.TestCase):
    def test_exact_shared_source_key_without_measurement(self):
        rot=(identity(),identity())
        first=same_source_address(723,rot)
        at_query=same_source_address(723,rot)
        self.assertEqual(first,at_query)
        self.assertAlmostEqual(sum(x*x for x in first[0]),1.0)
        self.assertEqual(len(source_features(723)),8)
        self.assertNotEqual(source_features(723),source_features(724))

    def test_source_int64_mod_avoids_float_position_alias(self):
        # The original Tensor implementation reduces ordinal modulo every
        # period while still integer, rather than rounding a huge timestamp.
        old=2**33+17
        self.assertEqual(source_features(old),source_features(old+7*13*29*59))

    def test_old_write_residual_and_identity_on_refusal(self):
        row=(.1,.2,.3,.4,.5,.6,.7,.8)
        key=(1.,0,0,0,0,0,0,0)
        self.assertEqual(gated_row_update(row,key,.9,.95,.8,False),row)
        new=gated_row_update(row,key,.9,.95,.8,True)
        self.assertAlmostEqual(new[0],.95*.1+.8*(.9-.95*.1))
        self.assertAlmostEqual(read_row(new,key),new[0])
        self.assertAlmostEqual(new[1],.95*row[1])

    def test_owner_history_permission_is_read_only(self):
        state=OwnerTimeline(5)
        with self.assertRaises(ValueError):
            state.permit_history(5,0)
        updated=state.accept(5,15).accept(5,18)
        self.assertTrue(updated.permit_history(5,15))
        self.assertEqual(updated.last,18)
        with self.assertRaises(ValueError):
            updated.permit_history(9,15)
        with self.assertRaises(ValueError):
            updated.permit_history(5,19)
        with self.assertRaises(ValueError):
            updated.accept(5,18)
        self.assertEqual(state.last,-1)

    def test_reject_non_source_target_input(self):
        with self.assertRaises(ValueError):
            source_features(-1)
        with self.assertRaises(ValueError):
            source_features(1.5)
        with self.assertRaises(ValueError):
            gated_row_update((0,)*8,(0,)*8,1.,1.2,.3,True)


if __name__=="__main__":
    unittest.main()
