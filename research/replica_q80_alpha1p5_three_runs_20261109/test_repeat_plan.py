"""确认两次重跑是独立seed0重复，所有输出保持隔离。"""
from pathlib import Path
import unittest
import prepare_experiment as p


class RepeatPlanTests(unittest.TestCase):
    def fixtures(self):
        return [dict(id=f'Replica/{s}/{m}/prune1/seed0_repeat1',dataset='Replica',scene=s,mode=m,
            prune=1,seed=0,repeat=1,frames=2000,environment={},
            command=[str(p.Q/'runtime/bin'/('replica_'+m)),'vocabulary',str(p.Q/'configs/Replica'/s/(m+'_camera.yaml')),
                str(p.Q/'configs/Replica'/s/(m+'_mapping.yaml')),'dataset','old_model','no_viewer','--selector-config',str(p.Q/'configs/selector_prune1.yaml')])
            for s in ['office0','office1','office2','office3','office4','room0','room1','room2'] for m in ['mono','rgbd']]

    def test_exact_twenty_four_new_repeats(self):
        jobs=p.planned_jobs(self.fixtures());self.assertEqual(len(jobs),24)
        self.assertEqual(len({x['id'] for x in jobs}),24)
        self.assertEqual([x['repeat'] for x in jobs],[2]*12+[3]*12)
        self.assertEqual({x['scene'] for x in jobs},set(p.SCENES))
        self.assertFalse({'room0','room1'} & {x['scene'] for x in jobs})

    def test_output_paths_and_original_preserved(self):
        prior=self.fixtures();original=repr(prior);jobs=p.planned_jobs(prior)
        self.assertEqual(repr(prior),original)
        for j in jobs:
            self.assertTrue(Path(j['output']).is_relative_to(p.R));self.assertEqual(j['command'][5],str(Path(j['output'])/'model'))
            self.assertEqual(j['state'],'held');self.assertNotIn('result',j)
            self.assertEqual(j['environment']['CARTGS_AUDIT_SEED'],'0')
            self.assertEqual(j['environment']['CARTGS_BATCH_FULL_TAIL_ONLY'],'1')
            self.assertTrue(j['command'][0].startswith(str(p.R)))

    def test_missing_or_wrong_seed_rejected(self):
        with self.assertRaises(AssertionError):p.planned_jobs(self.fixtures()[1:])
        jobs=self.fixtures();jobs[0]['seed']=1
        with self.assertRaises(AssertionError):p.planned_jobs(jobs)


if __name__=='__main__':unittest.main()
