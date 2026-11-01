import math
import unittest
from .math_reference import attention_reference


class VisualAttentionReference(unittest.TestCase):
    def test_zero_queries_mean_values(self):
        self.assertEqual(attention_reference([[0.,0.]],[[2.,3.],[4.,5.]],[[1.,3.],[5.,7.]]),[[3.,5.]])

    def test_softmax_shift_invariance(self):
        left=attention_reference([[1.,0.]],[[2.,3.],[4.,5.]],[[1.],[5.]])
        right=attention_reference([[1.,0.]],[[1002.,3.],[1004.,5.]],[[1.],[5.]])
        self.assertAlmostEqual(left[0][0],right[0][0],places=12)

    def test_extreme_logits_are_finite(self):
        result=attention_reference([[1000.,0.]],[[1000.,0.],[-1000.,0.]],[[2.],[-7.]])
        self.assertEqual(result,[[2.]])


if __name__=="__main__":
    unittest.main()
