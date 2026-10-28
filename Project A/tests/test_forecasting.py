import unittest
import numpy as np
import pandas as pd
from project_a.forecasting import seasonal_quantile_forecast,to_frame,validate_forecasts
class ForecastTests(unittest.TestCase):
 def test_quantile_ordering(self):
  x=np.arange(1,101,dtype=float)[None,:]; f=seasonal_quantile_forecast(x,80,3); self.assertTrue((f[.05]<=f[.95]).all())
 def test_schema_and_nonnegative(self):
  f=seasonal_quantile_forecast(np.ones((1,100)),80,2); z=to_frame(pd.DataFrame({'id':['a'],'item_id':['i'],'store_id':['s'],'dept_id':['FOODS_1']}),f,80); validate_forecasts(z)
