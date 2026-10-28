import unittest
import numpy as np
import pandas as pd
from project_a.simulate import simulate
class SimulationTests(unittest.TestCase):
 def test_reproducible_and_balanced(self):
  m=pd.DataFrame({'dept_id':['FOODS_1'],'id':['x'],'item_id':['x'],'store_id':['CA_1']});d=np.ones((1,8));q=np.ones((1,8));p=np.ones((1,8));a,_=simulate(m,d,q,p,np.ones(1),42,True);b,_=simulate(m,d,q,p,np.ones(1),42,True);self.assertEqual(a['total_cost'],b['total_cost']);self.assertEqual(a['fulfilled_units']+a['lost_units'],a['demand_units'])
 def test_fill_rate_bounds(self):
  m=pd.DataFrame({'dept_id':['FOODS_1'],'id':['x'],'item_id':['x'],'store_id':['CA_1']});r,_=simulate(m,np.ones((1,5)),np.ones((1,5)),np.ones((1,5)),np.ones(1),1,False);self.assertTrue(0<=r['fill_rate']<=1)
