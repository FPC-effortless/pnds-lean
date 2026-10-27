import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"foundation"/"v1"))
from latent_fusion_010_math import (
 source_delta_write,source_delta_memory_transfer,
 local_event_write_bypass,original_source_state_bytes)

class SourceEventMemoryMathOnly(unittest.TestCase):
    def test_exact_carried_state(self):
        self.assertEqual(original_source_state_bytes(),6800)
    def test_original_event_step_delta_identity(self):
        a,b,k,v,m=.997,.2,.75,.6,2.
        observed=source_delta_write(a,b,k,v,m)
        expected=local_event_write_bypass(a,b,k,v,m)
        self.assertAlmostEqual(observed-m,expected,places=12)
    def test_common_source_value_cancels(self):
        a,b,k,v=.99,.15,-.5,1.3
        first=source_delta_write(a,b,k,v,2.)
        second=source_delta_write(a,b,k,v,-1.)
        self.assertAlmostEqual(first-second,source_delta_memory_transfer(a,b,k,3.))
    def test_projected_memory_loss_positive_gates(self):
        a,b,k=.997,.2,1.
        self.assertLessEqual(abs(source_delta_memory_transfer(a,b,k,1.)),1.)
        self.assertGreaterEqual(abs(source_delta_memory_transfer(a,b,k,1.)),0.)
if __name__=="__main__":unittest.main()
