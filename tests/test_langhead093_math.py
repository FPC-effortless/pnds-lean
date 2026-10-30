"""Deterministic 32D Python reference tests; never model capability evidence."""
import random
import unittest
from foundation.v1.langhead093_math import score_head

class TiedDecoderMathTest(unittest.TestCase):
    def test_exact_equivalence_with_centering(self):
        for seed in (1,17,101):
            rng=random.Random(seed)
            e=[[rng.uniform(-.15,.15) for _ in range(32)] for _ in range(256)]
            for x in (0,17,255):
                a=score_head(e,x,.01,"euclidean")
                b=score_head(e,x,.01,"tied_dot_norm")
                self.assertLess(max(abs(u-v) for u,v in zip(a,b)),1e-12)
    def test_negative_control_and_split(self):
        rng=random.Random(23)
        e=[[rng.uniform(-.2,.2) for _ in range(32)] for _ in range(256)]
        a=score_head(e,27,.01,"tied_dot")
        b=score_head(e,27,.01,"shuffled_dot")
        self.assertGreater(max(abs(u-v) for u,v in zip(a,b)),.001)
        full=score_head(e,27,.01,"tied_dot_norm")
        norm=score_head(e,27,.01,"norm_only")
        self.assertLess(max(abs(u+v-w) for u,v,w in zip(a,norm,full)),1e-12)
    def test_missing_and_invalid_fails(self):
        e=[[0.]*32 for _ in range(256)]
        with self.assertRaises(ValueError):score_head(e,256,.01,"tied_dot")
        with self.assertRaises(ValueError):score_head(e,0,-1.,"tied_dot")
        with self.assertRaises(ValueError):score_head(e[:-1],0,.01,"tied_dot")
if __name__=="__main__":unittest.main()
