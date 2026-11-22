"""验证三遍完整性及最高PSNR模型的同次指标绑定。"""
import copy
import unittest
from compare_results import SCENES,MODES,select_best


class BestSelectionTests(unittest.TestCase):
    def fixture(self):
        return [dict(scene=s,mode=m,repeat=r,seed=0,psnr=30+r,gaussian_count=1000*r,
            checkpoint=f'{s}/{m}/{r}') for s in SCENES for m in MODES for r in [1,2,3]]

    def test_selects_same_run_count_and_path(self):
        rows=self.fixture();before=copy.deepcopy(rows);best,groups=select_best(rows)
        self.assertEqual(rows,before);self.assertEqual(len(best),16);self.assertEqual(len(groups),16)
        for row in best:
            self.assertEqual(row['repeat'],3);self.assertEqual(row['gaussian_count'],3000)
            self.assertTrue(row['checkpoint'].endswith('/3'))

    def test_tie_prefers_lower_repeat(self):
        rows=self.fixture()
        for row in rows:row['psnr']=30
        best,_=select_best(rows);self.assertTrue(all(x['repeat']==1 for x in best))

    def test_missing_duplicate_or_wrong_seed_rejected(self):
        rows=self.fixture()
        with self.assertRaises(AssertionError):select_best(rows[:-1])
        with self.assertRaises(AssertionError):select_best(rows+[rows[0]])
        rows[0]['seed']=1
        with self.assertRaises(AssertionError):select_best(rows)


if __name__=='__main__':unittest.main()
