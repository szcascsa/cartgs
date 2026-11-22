"""CPU验证健康/OOM重试、重启计数、普通错误、成功状态与检查范围。"""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import run_batch as b


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        (self.root/'source').mkdir();(self.root/'reports').mkdir()
        self.root_patch=patch.object(b,'R',self.root);self.root_patch.start()
        self.save_patch=patch.object(b.p,'save',lambda s:None);self.save_patch.start()
        self.memory_patch=patch.object(b.subprocess,'check_output',return_value='20000\n');self.memory_patch.start()
        self.sleep_patch=patch.object(b.time,'sleep',lambda _:None);self.sleep_patch.start()
        self.state={'protocol':{'max_health_retries':2}}
    def tearDown(self):
        for p in [self.sleep_patch,self.memory_patch,self.save_patch,self.root_patch]:p.stop()
        self.tmp.cleanup()
    def command(self,code):
        def argv(out):
            (out/'model').mkdir()
            return [sys.executable,'-c',"from pathlib import Path;import json,sys;out=Path("+repr(str(out))+');'+code]
        return argv
    def health(self,reason):
        return self.command("(out/'model/lod_health_failure.json').write_text(json.dumps({'error':"+repr(reason)+"}));sys.exit(2)")
    def test_health_retries_twice(self):
        job={}
        for n in range(3):
            self.assertIsNone(b.run_process(self.health('compensation_xyz_tail'),self.root/'job',job,self.state))
            self.assertEqual(job['health_failures'],n+1)
            self.assertEqual(job['state'],'retry_pending' if n<2 else 'skipped_health')
        self.assertEqual(len(job['attempt_history']),3)
    def test_zero_selection_retry(self):
        job={};b.run_process(self.health('consecutive_zero_selection_with_candidates'),self.root/'job',job,self.state)
        self.assertEqual(job['state'],'retry_pending')
    def test_scale_tail_retry(self):
        job={};b.run_process(self.health('compensation_scale_tail'),self.root/'job',job,self.state)
        self.assertEqual(job['state'],'retry_pending')
    def test_retry_count_survives_restart(self):
        job={'health_failures':2,'attempt_history':[]};job=json.loads(json.dumps(job))
        b.run_process(self.health('compensation_xyz_tail'),self.root/'job',job,self.state)
        self.assertEqual(job['state'],'skipped_health')
    def test_oom_retries_twice(self):
        job={};cmd=self.command("print('CUDA out of memory');sys.exit(1)")
        for _ in range(3):b.run_process(cmd,self.root/'job',job,self.state)
        self.assertEqual(job['state'],'skipped_oom');self.assertEqual(job['oom_failures'],3)
    def test_success_preserves_history(self):
        job={};out=b.run_process(self.command("print('finished')"),self.root/'job',job,self.state)
        self.assertTrue((out/'exit.json').exists());self.assertEqual(job['attempt_history'][0]['returncode'],0)
    def test_unknown_error_blocks(self):
        job={}
        with self.assertRaises(RuntimeError):b.run_process(self.command('sys.exit(3)'),self.root/'job',job,self.state)
        self.assertEqual(job['state'],'failed');self.assertNotIn('oom_failures',job)
    def test_cleanup_only_own_allowlist(self):
        model=self.root/'model';model.mkdir();keep=model/'selector.pt';keep.write_text('final')
        for name in ['step0_master.ply','step0_selector.pt','converted_selector_init.pt']:(model/name).write_text('initial')
        b.discard_initial_copies(model);self.assertEqual(keep.read_text(),'final');self.assertEqual(len(list(model.iterdir())),1)
        with self.assertRaises(ValueError):b.discard_initial_copies(self.root.parent/'outside')


if __name__=='__main__':unittest.main()
