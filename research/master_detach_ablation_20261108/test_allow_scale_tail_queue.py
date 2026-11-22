"""验证新增5次的边界、历史保留、首次成功停止与OOM独立上限。"""
import copy
import unittest
import test_queue
from prepare_allow_scale_tail_20261111 import requeue,TARGETS


class AllowScaleTailQueueTests(test_queue.QueueTests):
    def item(self):
        return dict(health_failures=20,oom_failures=0,max_training_attempts=25,max_health_retries=24,
                    attempt_history=[dict(output='historical_'+str(i)) for i in range(20)])

    def test_exactly_five_more_health_attempts(self):
        job=self.item()
        for n in range(5):
            test_queue.b.run_process(self.health('compensation_scale_tail'),self.root/'job',job,self.state)
            self.assertEqual(len(job['attempt_history']),21+n)
            self.assertEqual(job['health_failures'],21+n)
            self.assertEqual(job['state'],'retry_pending' if n<4 else 'skipped_health')
        self.assertEqual(len(list((self.root/'job').glob('attempt_*'))),5)

    def test_oom_still_stops_after_three_failures(self):
        job=self.item();cmd=self.command("print('CUDA out of memory');sys.exit(1)")
        for _ in range(3):test_queue.b.run_process(cmd,self.root/'job',job,self.state)
        self.assertEqual(job['oom_failures'],3);self.assertEqual(job['health_failures'],20)
        self.assertEqual(job['state'],'skipped_oom')

    def test_oom_cannot_exceed_total_opportunity_cap(self):
        job=self.item();job['attempt_history'].extend([dict(output='more_'+str(i)) for i in range(4)])
        test_queue.b.run_process(self.command("print('CUDA out of memory');sys.exit(1)"),self.root/'job',job,self.state)
        self.assertEqual(len(job['attempt_history']),25);self.assertEqual(job['state'],'skipped_oom')

    def test_success_immediately_stops_training_retries(self):
        job=self.item();test_queue.b.run_process(self.command("print('finished')"),self.root/'job',job,self.state)
        self.assertEqual(len(job['attempt_history']),21);self.assertEqual(job['attempt_history'][-1]['returncode'],0)
        self.assertEqual(job['health_failures'],20)

    def test_only_three_requeued_and_protocol_unchanged(self):
        state=dict(phase='finished',parallel_workers={},protocol={'max_health_retries':5,'max_oom_retries':2},
            jobs=[dict(id='complete',state='complete',trained_model='unchanged',evaluations=[dict(state='complete')]),
                *[dict(self.item(),id=name,state='skipped_health',evaluations=[]) for name in TARGETS]])
        before=copy.deepcopy(state);self.assertEqual(set(requeue(state)),TARGETS)
        self.assertEqual(state['protocol'],before['protocol']);self.assertEqual(state['jobs'][0],before['jobs'][0])
        for j,old in zip(state['jobs'][1:],before['jobs'][1:]):
            self.assertEqual(j['attempt_history'],old['attempt_history']);self.assertEqual(j['health_failures'],20)
            self.assertEqual(j['max_training_attempts'],25);self.assertEqual(j['max_health_retries'],24)
        self.assertEqual(state['final_audit_request'],'ALLOW_SCALE_TAIL_REQUEST.json')

    def test_per_job_extension_does_not_affect_unconfigured_job(self):
        self.assertFalse(test_queue.b.health_decision({'health_failures':20},5))
        self.assertTrue(test_queue.b.health_decision(self.item(),5))


if __name__=='__main__':unittest.main()
