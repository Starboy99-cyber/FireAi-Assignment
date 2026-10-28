import unittest
import numpy as np
from project_a.policy import lognormal_parameters,round_lead_time,allocate_capacity
class PolicyTests(unittest.TestCase):
 def test_positive_lead_times(self):
  self.assertTrue((round_lead_time([.01,1.2])>0).all()); self.assertEqual(len(lognormal_parameters(5,.4)),2)
 def test_capacity_allocation(self):
  x=allocate_capacity(np.array([5.,5.]),np.array([1.,1.]),np.array([2.,1.]),3); self.assertLessEqual(x.sum(),3)
