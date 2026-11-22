"""CPU验证额外5次对应总6次、历史次数保留、只恢复健康失败任务。"""
import copy
import unittest
import test_queue
from extend_health_retries import update_state


class ExtendedRetryTests(test_queue.QueueTests):
    def test_extra_five_allows_six_total_attempts(self):
        self.state['protocol']['max_health_retries']=5
        job={}
        for n in range(6):
            self.assertIsNone(test_queue.b.run_process(self.health('compensation_scale_tail'),
                              self.root/'job',job,self.state))
            self.assertEqual(job['health_failures'],n+1)
            self.assertEqual(job['state'],'retry_pending' if n<5 else 'skipped_health')
        self.assertEqual(len(job['attempt_history']),6)

    def test_consumed_three_failures_survive_extension_and_restart(self):
        self.state['protocol']['max_health_retries']=5
        job={'health_failures':3,'attempt_history':[]}
        for n in range(3):
            test_queue.b.run_process(self.health('compensation_scale_tail'),self.root/'job',job,self.state)
            self.assertEqual(job['health_failures'],4+n)
            self.assertEqual(job['state'],'retry_pending' if n<2 else 'skipped_health')

    def test_only_health_jobs_requeued_without_reset(self):
        history=[{'output':'attempt_01'},{'output':'attempt_02'},{'output':'attempt_03'}]
        state=dict(protocol={'max_health_retries':2,'max_oom_retries':2},jobs=[
            dict(id='health',state='skipped_health',health_failures=3,oom_failures=0,attempt_history=history),
            dict(id='oom',state='skipped_oom',health_failures=0,oom_failures=3),
            dict(id='complete',state='complete',trained_model='model'),
            dict(id='exhausted',state='skipped_health',health_failures=6)])
        before=copy.deepcopy(state)
        self.assertEqual(update_state(state),['health'])
        self.assertEqual(state['protocol']['max_health_retries'],5)
        self.assertEqual(state['protocol']['max_oom_retries'],2)
        self.assertEqual(state['jobs'][0]['health_failures'],3)
        self.assertEqual(state['jobs'][0]['attempt_history'],history)
        self.assertEqual(state['jobs'][1:],before['jobs'][1:])


if __name__=='__main__':unittest.main()
