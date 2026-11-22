"""验证只续评目标场景、复用已完成结果，以及失败时恢复暂停。"""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import tempfile
import unittest
import resume_office4_eval as runner


class ResumeTests(unittest.TestCase):
    def exercise(self, failure=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'reports').mkdir()
            (root/'HOLD').write_text('用户暂停\n')
            state = dict(phase='held', jobs=[
                dict(id='Replica/office3/rgbd', state='complete', evaluations=[]),
                dict(id='Replica/office4/rgbd', state='trained', trained_model='saved_model', evaluations=[
                    dict(scope='global', ratio=.1, state='complete', result={'psnr': 30}),
                    dict(scope='view_ratio', ratio=.1, state='pending')]),
                dict(id='Replica/room0/mono', state='pending', evaluations=[])])
            original = copy.deepcopy(state)
            def write(path, value):
                path.write_text(json.dumps(value))
            def save(value):
                write(root/'LOD_PIPELINE.json', value)
            save(state)
            calls = []
            def evaluate(job, item, value, output):
                calls.append((job['id'], item['scope'], item['ratio']))
                self.assertFalse((root/'HOLD').exists())
                if failure:
                    raise RuntimeError('模拟评价故障')
                item.update(state='complete', result={'psnr': 31})
            b = SimpleNamespace(R=root, p=SimpleNamespace(
                read=lambda path: json.loads(path.read_text()), write=write, save=save,
                identity=lambda pid: {'pid': pid}), evaluate=evaluate, report=lambda state: None)
            if failure:
                with self.assertRaisesRegex(RuntimeError, '模拟评价故障'):
                    runner.run(b)
            else:
                runner.run(b)
            after = json.loads((root/'LOD_PIPELINE.json').read_text())
            self.assertTrue((root/'HOLD').exists())
            self.assertEqual(calls, [('Replica/office4/rgbd', 'view_ratio', .1)])
            self.assertEqual(after['jobs'][0], original['jobs'][0])
            self.assertEqual(after['jobs'][2], original['jobs'][2])
            self.assertEqual(after['jobs'][1]['evaluations'][0], original['jobs'][1]['evaluations'][0])
            self.assertEqual(after['phase'], 'blocked' if failure else 'held_for_li05_controls')
            if not failure:
                self.assertEqual(after['jobs'][1]['state'], 'complete')

    def test_resume_only_pending_evaluation(self):
        self.exercise()

    def test_failure_restores_hold(self):
        self.exercise(failure=True)


if __name__ == '__main__':
    unittest.main()
