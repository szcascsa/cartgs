"""确保只恢复明确的六组VECtor，并按修正后的双目指标选赢家。"""
import copy
import unittest
from prepare_vector import validate_jobs


def fixture():
    jobs = []
    for scene in ['corner-slow', 'corridors-dolly', 'robot-normal']:
        for mode in ['mono', 'stereo']:
            candidates = [dict(id=f'VECtor/{scene}/{mode}/repeat{i}', repeat=i, train_state='complete',
                eval_state='complete', result=dict(trajectory_psnr=20+i, correct_scale=mode == 'mono',
                **({'uncorrected_trajectory_psnr': 40-i} if mode == 'stereo' else {}))) for i in [1, 2, 3]]
            jobs.append(dict(id=f'VECtor/{scene}/{mode}', dataset='VECtor', scene=scene, mode=mode,
                candidates=candidates, winner=candidates[2]['id'], full_psnr=23, target_steps=100,
                full_job={'result': {'online_steps': 100}}))
    return jobs


class ScopeTests(unittest.TestCase):
    def test_exact_six_and_corrected_stereo_winners(self):
        validate_jobs(fixture())

    def test_duplicate_scene_rejected(self):
        jobs = fixture()
        jobs[-1] = copy.deepcopy(jobs[0])
        with self.assertRaises(AssertionError):
            validate_jobs(jobs)

    def test_other_dataset_rejected(self):
        jobs = fixture()
        jobs[0]['dataset'] = 'TUM_RGBD'
        with self.assertRaises(AssertionError):
            validate_jobs(jobs)

    def test_uncorrected_stereo_winner_rejected(self):
        jobs = fixture()
        jobs[1]['winner'] = jobs[1]['candidates'][0]['id']
        with self.assertRaises(AssertionError):
            validate_jobs(jobs)


if __name__ == '__main__':
    unittest.main()
