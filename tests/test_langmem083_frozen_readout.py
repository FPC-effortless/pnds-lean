import importlib.util,unittest,sys
from pathlib import Path
p=Path(__file__).resolve().parents[1]/"foundation/v1/langmem083_frozen_readout.py"
spec=importlib.util.spec_from_file_location("langmem083_frozen_readout",p)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m
spec.loader.exec_module(m)

class FrozenDecoderMath(unittest.TestCase):
    def test_frozen_model_total_and_head_counts(self):
        self.assertEqual((m.INPUT,m.TRAINABLE,m.DEPLOYED,m.EXTRA_FP32_BYTES),
                         (20,9120,508747,36480))
    def test_exact_onehot_feature_age_and_source_clock(self):
        for a in m.AGES:
            self.assertEqual(sum(m.age_onehot(a)),1)
            v=m.feature([0.1]*8,[0.2]*8,a)
            self.assertEqual(len(v),20)
            self.assertEqual(v[-4:].index(1),m.AGES.index(a))
            expected=(128-a if 128-a>=4 else None)
            self.assertEqual(m.historical_anchor(128,a),expected)
        with self.assertRaises(ValueError):m.age_onehot(7)
    def test_earlier_anchor_only_causal(self):
        for t in range(128,640,4):
            for age in m.AGES:
                a=m.historical_anchor(t,age)
                if a is not None:self.assertTrue(4<=a<t)
        with self.assertRaises(ValueError):m.historical_anchor(640,8)
