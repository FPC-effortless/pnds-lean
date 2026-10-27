import importlib.util
import math
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('history_query016',Path(__file__).resolve().parents[1]/'foundation/v1/history_query016.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class LossTests(unittest.TestCase):
    def test_mask_and_boundaries(self):
        self.assertEqual(m.historical_loss([0.,1.,5.],[0.,0.,math.nan],[True,True,False]),.25)
        self.assertEqual(m.historical_loss([9.],[math.nan],[False]),0.)
        self.assertEqual(m.historical_loss([2.],[0.],[True]),1.5)
    def test_forecast_mask(self):
        self.assertAlmostEqual(m.forecast_loss([0.,3.],[1.,1.],[0.,math.nan],[True,False]),.4*math.log(2*math.pi))
        with self.assertRaises(ValueError):m.forecast_loss([0.],[0.],[0.],[True])
if __name__=='__main__':unittest.main()
