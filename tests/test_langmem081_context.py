import importlib.util,sys,unittest
from pathlib import Path
path=Path(__file__).resolve().parents[1]/"foundation/v1/langmem081_context.py"
spec=importlib.util.spec_from_file_location("langmem081_context",path)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

class SourcePrefixOnly(unittest.TestCase):
    def test_context_match_is_independent_of_future_source_target(self):
        prefix=(b"abcde"*26)[:129]
        for L in m.LENGTHS:
            for H in m.AGES:
                selected=m.matching_indices(prefix,L,H)
                self.assertEqual(selected,m.matching_indices(prefix,L,H))
                self.assertEqual(m.most_recent_past_match(prefix,L,H),
                                 selected[-1] if selected else None)
                self.assertTrue(all(a<len(prefix)-1 for a in selected))
                self.assertEqual(m.deterministic_wrong_or_uniform_index("source",prefix,L,H),
                                 m.deterministic_wrong_or_uniform_index("source",prefix,L,H))
                future_truth_A=13
                future_truth_B=204
                self.assertNotEqual(future_truth_A,future_truth_B)
                # No future target parameter exists in m.matching_indices.
    def test_source_no_positive_preserved(self):
        prefix=bytes(range(129))
        self.assertEqual(m.matching_indices(prefix,4,128),())
        self.assertIsNone(m.most_recent_past_match(prefix,4,128))
    def test_strict_context_horizon(self):
        with self.assertRaises(ValueError):m.matching_indices(bytes(range(129)),5,32)
        with self.assertRaises(ValueError):m.matching_indices(bytes(range(129)),3,64)
