import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"foundation"/"v1"))
from latent_fusion_002_math import (multirate_coefficients,slow_update,
                                    initialize_memory_gates,sigmoid,belief_query,contract)

class FixedStateMath(unittest.TestCase):
    def test_retained_bytes(self):
        self.assertEqual(contract(),dict(state_float_values=1696,
                        occupancy_flags=16,nominal_fp32_bytes=6800))
    def test_rates(self):
        rates=multirate_coefficients()
        self.assertEqual(len(rates),80)
        self.assertAlmostEqual(rates[0],.90,places=12)
        self.assertAlmostEqual(rates[-1],.99999,places=12)
        self.assertGreater(rates[-1]**512,.99)
    def test_gates(self):
        gates=initialize_memory_gates()
        self.assertAlmostEqual(sigmoid(gates["retention_logit"]),.995,places=12)
        self.assertAlmostEqual(sigmoid(gates["write_logit"]),.1,places=12)
    def test_slow_difference(self):
        a=slow_update([2.],[0.],[.9])[0]
        b=slow_update([3.],[0.],[.9])[0]
        self.assertAlmostEqual(a-b,.9*(2.-3.),places=12)
    def test_query(self):
        self.assertEqual(belief_query([1.,2.],[.5,-.5]),[1.5,1.5])
if __name__=="__main__":unittest.main()
