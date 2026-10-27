import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"foundation"/"v1"))
from latent_fusion_007_math import contract,label_free_query

class HistoricQueryCounts(unittest.TestCase):
    def test_same_original_associative_capacity(self):
        self.assertEqual(contract()["matrix_cells"],256)
    def test_same_6800_byte_stream(self):
        x=contract()
        self.assertEqual(x["carried_fp32_values"],1696)
        self.assertEqual(x["nominal_bytes"],6800)
    def test_99390_original_trainable_params(self):
        x=contract()
        self.assertEqual(x["added_query_parameters"],129)
        self.assertEqual(x["candidate_live_parameters"],99390)
    def test_abstract_source_label_absent(self):
        called=[]
        def read(s,c):
            called.append((s,c))
            return s+c
        self.assertEqual(label_free_query(read,4,5),9)
        self.assertEqual(called,[(4,5)])
if __name__=="__main__":unittest.main()
