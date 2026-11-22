"""验证只重新排入三组，不重置次数，不改完成任务或训练协议。"""
import copy
import unittest
from rerun_final_failures_20261111 import requeue,TARGETS


class RerunTests(unittest.TestCase):
    def setUp(self):
        self.state=dict(phase='finished',parallel_workers={},protocol=dict(max_health_retries=5),jobs=[
            dict(id='completed',state='complete',trained_model='immutable',evaluations=[dict(state='complete')]),
            *[dict(id=name,state='skipped_health',health_failures=6,oom_failures=0,
                   attempt_history=[dict(output='attempt_'+str(i)) for i in range(1,7)],evaluations=[])
              for name in sorted(TARGETS)]])
    def test_only_failed_three_and_preserve_history(self):
        before=copy.deepcopy(self.state)
        self.assertEqual(set(requeue(self.state)),TARGETS)
        self.assertEqual(self.state['jobs'][0],before['jobs'][0])
        self.assertEqual(self.state['protocol'],before['protocol'])
        for job,old in zip(self.state['jobs'][1:],before['jobs'][1:]):
            self.assertEqual(job['state'],'retry_pending')
            self.assertEqual(job['health_failures'],6)
            self.assertEqual(job['attempt_history'],old['attempt_history'])
            self.assertEqual(job['manual_rerun_history'][-1]['next_attempt'],7)
    def test_active_queue_rejected(self):
        self.state['phase']='parallel_running'
        with self.assertRaises(AssertionError):requeue(self.state)
    def test_changed_failure_scope_rejected(self):
        self.state['jobs'][1]['id']='unexpected'
        with self.assertRaises(AssertionError):requeue(self.state)
    def test_completed_model_on_failed_job_rejected(self):
        self.state['jobs'][1]['trained_model']='unexpected_model'
        with self.assertRaises(AssertionError):requeue(self.state)


if __name__=='__main__':unittest.main()
