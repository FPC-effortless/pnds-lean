import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('reference003',Path(__file__).resolve().parents[1]/'foundation/v1/lacuna_retrieval003.py')
ref=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


class RetrievalReference(unittest.TestCase):
    def test_attention_normalized_and_constant_value(self):
        v=[float(i) for i in range(8)]
        result,weights=ref.attention([[1.]*8,[-1.]*8],[v,v],[1.]*8)
        self.assertAlmostEqual(sum(weights),1.)
        self.assertEqual(result,v)

    def test_match_norm_zero_and_transport(self):
        self.assertEqual(ref.match_norm([0.]*8,[1.]*8),[0.]*8)
        self.assertEqual(ref.match_norm([1.]*8,[2.]*8),[2.]*8)
        self.assertEqual(ref.transported_read([[1.]*8,[-1.]*8],[[1.]*8,[1.]*8],[1.]*8),[0.]*8)
        self.assertEqual(8*8*4,8+248)


if __name__=='__main__': unittest.main()
