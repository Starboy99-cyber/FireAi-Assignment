import unittest
import pandas as pd
from project_a.reconciliation import reconcile_scenarios
class ReconciliationTests(unittest.TestCase):
 def test_scenarios_aggregate(self):
  d={'id':['a','b'],'item_id':['i','i'],'store_id':['s1','s2'],'dept_id':['FOODS_1']*2,'target_day':[1,1]};d.update({f'q{i:02d}':[1.,2.] for i in range(5,100,5)});sc,_=reconcile_scenarios(pd.DataFrame(d),draws=5);self.assertEqual(len(sc),5);self.assertTrue((sc.dc_demand>=3).all())
