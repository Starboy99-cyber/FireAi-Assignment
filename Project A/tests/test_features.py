import unittest
import pandas as pd
from project_a.features import build_lag_features
class FeatureTests(unittest.TestCase):
 def test_lags_shifted(self):
  x=build_lag_features(pd.DataFrame({'id':['a']*4,'day_num':[1,2,3,4],'sales':[1,2,3,4]})); self.assertEqual(x.loc[2,'lag_1'],2)
 def test_no_current_target_in_roll(self):
  x=build_lag_features(pd.DataFrame({'id':['a']*3,'day_num':[1,2,3],'sales':[1,2,100]})); self.assertLess(x.loc[2,'roll_mean_7'],10)
