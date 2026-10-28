import unittest
import numpy as np
import pandas as pd
from src.transformations import geometric_adstock, hill_saturation
from src.validation import chronological_split
from src.pipeline import _optimize, _metrics

class CoreTests(unittest.TestCase):
 def test_adstock_is_causal(self): self.assertTrue(np.allclose(geometric_adstock(np.array([1.,2.,3.]),.5),[1,2.5,4.25]))
 def test_adstock_rejects_invalid_parameter(self):
  with self.assertRaises(ValueError): geometric_adstock(np.array([1.]),1.0)
 def test_hill_is_bounded(self):
  x=hill_saturation(np.array([0.,10.,100.]),10.); self.assertEqual(x[0],0); self.assertLess(x[1],1); self.assertLess(x[2],1)
 def test_revenue_construction(self):
  d=pd.DataFrame({"conversions":[100.,250.],"revenue_per_conversion":[2.,3.]}); self.assertEqual(float((d.conversions*d.revenue_per_conversion).sum()),950.)
 def test_split_has_no_outcome_leakage(self):
  d=pd.DataFrame({"time":pd.date_range("2020-01-06",periods=70,freq="7D").repeat(2),"geo":["a","b"]*70}); tr,ho,start=chronological_split(d,None); self.assertLess(tr.time.max(),start); self.assertEqual(ho.time.min(),start)
 def test_budget_bounds_and_sum(self):
  baseline=np.array([10.,20.,30.,40.,50.])*1_000_000; draws=np.tile(np.array([[1.],[2.],[3.],[4.],[5.]])*1_000_000,(1,10)); result=_optimize(baseline,draws,float(baseline.sum())); self.assertAlmostEqual(result["spend"].sum(),baseline.sum(),places=3); self.assertTrue(np.all(result["spend"]>=.7*baseline)); self.assertTrue(np.all(result["spend"]<=1.3*baseline))
 def test_infeasible_budget_is_reported(self):
  b=np.ones(5)*1_000_000; self.assertIsNone(_optimize(b,np.ones((5,2)),2_000_000))
 def test_forecast_metric(self):
  m=_metrics(np.array([10.,20.]),np.array([[9.,11.],[19.,21.]])); self.assertAlmostEqual(m["MAE"],0.)

if __name__ == "__main__": unittest.main()
