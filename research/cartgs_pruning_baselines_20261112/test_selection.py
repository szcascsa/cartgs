"""严格预算、排序稳定性与跨Ratio嵌套的CPU验收。"""
import unittest
import numpy as np
from common import budget, order_points, RATIOS


class SelectionTests(unittest.TestCase):
    def test_decimal_budget(self):
        self.assertEqual(budget('0.7',90),63)
        self.assertEqual(budget('0.3',10),3)
        self.assertEqual(budget('0.01',4995),49)
        self.assertEqual(budget('0.01',8167),81)

    def test_ties_keep_original_rows(self):
        np.testing.assert_array_equal(order_points('opacity',np.array([.1,.8,.8,1.,0.])),[3,1,2,0,4])

    def test_exact_nested_and_repeatable(self):
        for count in [0,1,17,1000,139948]:
            opacity=np.linspace(0,1,count,dtype=np.float32)
            for method in ['random','opacity']:
                ranked=order_points(method,opacity)
                np.testing.assert_array_equal(ranked,order_points(method,opacity))
                self.assertEqual(len(np.unique(ranked)),count)
                previous=set()
                for ratio in RATIOS+['1.0']:
                    k=budget(ratio,count);selected=set(ranked[:k].tolist())
                    self.assertEqual(len(selected),k);self.assertTrue(previous<=selected)
                    previous=selected


if __name__=='__main__':unittest.main()
