import importlib.util
from pathlib import Path
import unittest
p=Path(__file__).resolve().parents[1]/"foundation/v1/langscale070_ffn.py"
s=importlib.util.spec_from_file_location("langscale070_ffn",p)
m=importlib.util.module_from_spec(s)
import sys
sys.modules[s.name]=m
s.loader.exec_module(m)


class FFNMath(unittest.TestCase):
    def test_registered_all_axes(self):
        self.assertEqual([m.registered(w).trainable for w in (150,455,1067)],
                         [250133,499623,1000239])
        self.assertTrue(all(m.registered(w).upper_width==48 for w in (150,455,1067)))

    def test_exact_linear_growth(self):
        self.assertEqual(m.counts(455).trainable-m.counts(150).trainable,818*(455-150))
        self.assertEqual(m.counts(1067).trainable-m.counts(455).trainable,818*(1067-455))

    def test_unsupported_and_bad_shapes(self):
        for w in (0,-1,2.5,True):
            with self.assertRaises(ValueError):
                m.counts(w)
        for w in (80,151,2000):
            with self.assertRaises(ValueError):
                m.registered(w)
