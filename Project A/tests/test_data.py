import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from project_a.data import resolve_input

class DataTests(unittest.TestCase):
 def test_resolve_extension(self):
  self.assertEqual(resolve_input(Path(__file__)),Path(__file__).resolve())
 def test_nonnegative_shape(self):
  x=np.array([[0,1],[2,0]]); self.assertEqual(int((x<0).sum()),0)
 def test_unique_series_day(self):
  x=pd.DataFrame({'id':['a','a'],'day_num':[1,2]}); self.assertFalse(x.duplicated(['id','day_num']).any())
