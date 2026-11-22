"""CPU验证重启恢复：完成结果不重跑、旧现场保留、健康与OOM计数不变。"""
from pathlib import Path
import copy
import json
import tempfile
import unittest
from unittest.mock import patch
import recover_server_reboot as recovery


class RebootTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.patches=[patch.object(recovery,'R',self.root),
                      patch.object(recovery.supervisor,'alive',return_value=False)]
        for p in self.patches:p.start()
        self.out=self.root/'attempt_01';self.out.mkdir()
        self.item=dict(state='running',process=dict(pid=1,boot_id='old'),attempt=str(self.out),
                       health_failures=2,oom_failures=1,attempt_history=[])
    def tearDown(self):
        for p in self.patches:p.stop()
        self.tmp.cleanup()
    def test_reboot_preserves_counters_and_artifacts(self):
        log=self.out/'train.log';log.write_bytes(b'partial\x00')
        recovery.recover_item(self.item,'old','new')
        self.assertEqual(self.item['state'],'retry_pending')
        self.assertEqual((self.item['health_failures'],self.item['oom_failures']),(2,1))
        self.assertEqual(log.read_bytes(),b'partial\x00')
        self.assertEqual(len(self.item['attempt_history']),1)
    def test_same_boot_is_rejected(self):
        with self.assertRaises(AssertionError):recovery.recover_item(self.item,'old','old')
    def test_live_process_is_rejected(self):
        with patch.object(recovery.supervisor,'alive',return_value=True):
            with self.assertRaises(AssertionError):recovery.recover_item(self.item,'old','new')
    def test_explicit_failure_cannot_be_reclassified_as_reboot(self):
        (self.out/'exit.json').write_text('{"returncode":1}')
        with self.assertRaises(AssertionError):recovery.recover_item(self.item,'old','new')
    def test_health_failure_cannot_be_reclassified(self):
        (self.out/'model').mkdir();(self.out/'model/lod_health_failure.json').write_text('{}')
        with self.assertRaises(AssertionError):recovery.recover_item(self.item,'old','new')
    def test_duplicate_recovery_is_rejected(self):
        recovery.recover_item(self.item,'old','new')
        with self.assertRaises(AssertionError):recovery.recover_item(self.item,'old','new')
    def test_eval_recovery_retains_model_and_completed_ratios(self):
        complete=dict(state='complete',scope='global',ratio=.01,result=dict(psnr=30))
        model=self.root/'model';model.mkdir()
        job=dict(id='Replica/a/rgbd',state='trained',trained_model=str(model),evaluations=[complete,self.item])
        worker=self.root/'worker.json';worker.write_text(json.dumps(dict(jobs=[job],worker_phase='running')))
        state=dict(jobs=[copy.deepcopy(job)],parallel_workers=dict(eval=dict(
                   job=job['id'],kind='eval',path=str(worker),process=dict(pid=2,boot_id='old'))))
        recovery.recover_workers(state,'new');restored=state['jobs'][0]
        self.assertEqual(restored['trained_model'],str(model))
        self.assertEqual(restored['evaluations'][0],complete)
        self.assertEqual(restored['evaluations'][1]['state'],'retry_pending')
        self.assertEqual(state['parallel_workers'],{})


if __name__=='__main__':unittest.main()
