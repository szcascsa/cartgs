"""仅CPU检查32组对照绑定、单一配置差异和禁止提前启动。"""
from pathlib import Path
from collections import Counter
import json
import tempfile
import unittest
from unittest.mock import patch
import run_batch as b
import supervisor

R = Path(__file__).resolve().parent
B = Path('/home/prmi/zwj/realLOD/baseline/dual_view_li_best_full_20261107')
V = B.parent/'dual_view_li_vector_best_full_20261107'


def read(path):
    return json.loads(path.read_text())


class AblationTests(unittest.TestCase):
    def test_scope_and_pending(self):
        state = read(R/'LOD_PIPELINE.json')
        self.assertEqual(state['phase'], 'prepared_held')
        self.assertEqual(len(state['jobs']), 32)
        self.assertEqual(Counter(j['dataset'] for j in state['jobs']),
                         {'Replica': 16, 'TUM_RGBD': 10, 'VECtor': 6})
        self.assertTrue(all(j['state'] == 'pending' and not j['attempt_history']
                            and not j['evaluations'] for j in state['jobs']))
        self.assertEqual(sum(j['target_steps'] for j in state['jobs']), 637377)

    def test_paired_full_initialization(self):
        reference = {j['id']: j for root in [B, V] for j in read(root/'LOD_PIPELINE.json')['jobs']}
        for job in read(R/'LOD_PIPELINE.json')['jobs']:
            previous = reference[job['id']]
            for key in ['full_job', 'full_evaluation', 'target_steps', 'full_psnr']:
                self.assertEqual(job[key], previous[key], (job['id'], key))
            self.assertEqual(job['baseline_model'], previous['trained_model'])
            self.assertNotEqual(job['full_job']['result']['checkpoint'], job['baseline_model'])
            self.assertTrue(Path(job['full_job']['result']['checkpoint']).is_dir())
            for key in ['camera_config', 'mapping_config']:
                self.assertEqual(Path(job[key]).read_bytes(), Path(previous[key]).read_bytes())

    def test_only_detach_config_changed(self):
        current = (R/'configs/dual_view_li.yaml').read_text()
        previous = (B/'configs/dual_view_li.yaml').read_text()
        self.assertEqual(current.replace('Selector.detach_gaussian_parameters: 1',
                                         'Selector.detach_gaussian_parameters: 0'), previous)

    def test_runner_honors_hold(self):
        self.assertTrue((R/'HOLD').exists())
        self.assertFalse((R/'START_AUTHORIZED.json').exists())
        with self.assertRaises(AssertionError):
            b.main()

    def test_supervisor_honors_hold(self):
        with self.assertRaises(AssertionError):
            supervisor.main()

    def test_legacy_entry_honors_hold(self):
        with self.assertRaises(AssertionError):
            b.p.main()

    def test_evaluation_scope(self):
        cases = b.cases()
        self.assertEqual(len(cases), 23)
        self.assertEqual(Counter(j['scope'] for j in cases), {'global': 11, 'view_ratio': 11, 'full': 1})

    def test_frozen_verification_rejects_invalid_state(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            model = out/'model'
            ply = model/'1_shutdown/ply'
            ply.mkdir(parents=True)
            report = dict(passed=True, topology_unchanged=True, master_xyz_changed=False, steps=6)
            stage = {k: 6 for k in ['lod_steps', 'selector_updates', 'full_training_renders',
                     'full_training_backwards', 'lod_training_renders', 'lod_training_backwards', 'optimizer_commits']}
            stage.update(master_updates=0, master_frozen_in_lod=True, lod_nonfinite=0,
                         ratio_steps=[1]*6, ratios=b.p.TRAIN)
            for file, value in [(model/'training_verified.json', report),
                                (ply/'real_lod_state.json', stage), (out/'manifest.json', {'input_sha256': {}})]:
                file.write_text(json.dumps(value))
            self.assertEqual(b.original_verify(out, 6), ply)
            for key, value in [('master_updates', 1), ('master_frozen_in_lod', False)]:
                changed = dict(stage, **{key: value})
                (ply/'real_lod_state.json').write_text(json.dumps(changed))
                with self.assertRaises(AssertionError):
                    b.original_verify(out, 6)
            (ply/'real_lod_state.json').write_text(json.dumps(stage))
            for key in ['all_six_master_fields_bitwise_frozen', 'master_gradients_absent',
                        'master_adam_state_absent', 'anchor_feature_adam_present']:
                report[key] = False
            (model/'training_verified.json').write_text(json.dumps(report))
            with self.assertRaises(AssertionError):
                b.verify_training(out, 6)


if __name__ == '__main__':
    unittest.main()
