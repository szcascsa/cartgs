"""CPU验证训练/评价重叠、单一主状态写入和安全接管。"""
from pathlib import Path
from unittest.mock import patch
import copy
import json
import subprocess
import sys
import tempfile
import time
import unittest
import parallel_queue as q


def job(name,state='pending'):
    return dict(id=name,dataset=name.split('/')[0],mode=name.split('/')[-1],state=state,evaluations=[])


class ParallelTests(unittest.TestCase):
    def setUp(self):
        self.state=dict(jobs=[job('Replica/a/mono','trained'),job('Replica/b/mono')],
                        parallel_workers={},validated_modes=['Replica/mono'],protocol={})

    def test_train_and_eval_choose_distinct_jobs(self):
        self.assertEqual(q.next_work(self.state,'train')[1]['id'],'Replica/b/mono')
        self.assertEqual(q.next_work(self.state,'eval')[1]['id'],'Replica/a/mono')

    def test_eval_requires_completed_training(self):
        self.state['jobs'][0]['state']='running'
        self.assertIsNone(q.next_work(self.state,'eval'))

    def test_busy_job_not_launched_twice(self):
        self.state['parallel_workers']['eval']=dict(job='Replica/a/mono')
        self.assertIsNone(q.next_work(self.state,'eval'))

    def test_new_mode_requires_smoke(self):
        self.state['validated_modes']=[]
        self.assertEqual(q.next_work(self.state,'train')[0],'validate')

    def test_exclusive_stereo_waits_for_evaluation(self):
        self.state['jobs'][1]=job(q.EXCLUSIVE)
        self.state['parallel_workers']['eval']=dict(job='Replica/a/mono')
        self.assertIsNone(q.next_work(self.state,'train'))
        self.state['parallel_workers']={}
        self.assertEqual(q.next_work(self.state,'train')[1]['id'],q.EXCLUSIVE)

    def test_exclusive_stereo_blocks_other_lane(self):
        self.state['parallel_workers']['train']=dict(job=q.EXCLUSIVE)
        self.assertIsNone(q.next_work(self.state,'eval'))

    def collect(self,kind,phase='done',alive=False,native=False):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'receipt.json'
            j=copy.deepcopy(self.state['jobs'][0]);j['state']='complete'
            j['process']={'pid':20};j['evaluations']=[dict(state='complete',result=dict(psnr=30))]
            path.write_text(json.dumps(dict(jobs=[j],worker_phase=phase,worker_error='fixture')))
            self.state['parallel_workers']['eval']=dict(job=j['id'],kind=kind,path=str(path),process={'pid':10})
            with patch.object(q.supervisor,'alive',side_effect=lambda p:alive if p.get('pid')==10 else native):
                return q.collect(self.state,'eval')

    def test_merge_only_own_job_and_keep_other_progress(self):
        self.state['jobs'][1].update(state='running',attempt='ongoing',health_failures=2)
        other=copy.deepcopy(self.state['jobs'][1]);self.assertIsNone(self.collect('eval'))
        self.assertEqual(self.state['jobs'][1],other)
        self.assertEqual(self.state['jobs'][0]['evaluations'][0]['result']['psnr'],30)
        self.assertFalse(self.state['parallel_workers'])

    def test_live_worker_keeps_reserved_slot(self):
        self.assertIsNone(self.collect('eval',alive=True))
        self.assertIn('eval',self.state['parallel_workers'])

    def test_missing_completion_is_error(self):
        self.assertEqual(self.collect('eval',phase='running'),'fixture')

    def test_orphan_native_blocks_relaunch(self):
        with self.assertRaises(AssertionError):self.collect('eval',native=True)
        self.assertIn('eval',self.state['parallel_workers'])

    def test_worker_saves_private_receipt_without_touching_main_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            r=Path(directory);path=r/'receipt.json';main=r/'LOD_PIPELINE.json';main.write_text('immutable')
            j=job('Replica/a/mono','trained');j['trained_model']=str(r);j['evaluations']=[dict(scope='full',ratio=1.,state='pending')]
            path.write_text(json.dumps(dict(jobs=[j],protocol={})))
            def evaluate(job,item,state,root):
                item.update(state='complete',result=dict(psnr=30));q.p.save(state)
            with patch.object(q.b,'evaluate',side_effect=evaluate),patch.object(q.p,'save'):
                q.worker('eval',path)
            self.assertEqual(main.read_text(),'immutable')
            result=q.p.read(path);self.assertEqual(result['worker_phase'],'done')
            self.assertEqual(result['jobs'][0]['state'],'complete')

    def test_recovery_attaches_live_lane_without_relaunch(self):
        self.state['parallel_workers']['train']=dict(job='Replica/b/mono',process={'pid':10})
        with patch.object(q.supervisor,'alive',return_value=True),patch.object(q,'collect') as collect,patch.object(q.p,'save'):
            self.assertTrue(q.supervisor.recover(self.state))
            collect.assert_not_called()
        self.assertIn('train',self.state['parallel_workers'])

    def test_recovery_blocks_unknown_worker_exit(self):
        self.state['parallel_workers']['train']=dict(job='Replica/b/mono',process={'pid':10})
        with patch.object(q.supervisor,'alive',return_value=False),patch.object(q,'collect',return_value='unknown error'),patch.object(q.p,'save'):
            self.assertFalse(q.supervisor.recover(self.state))

    def test_two_real_cpu_processes_overlap_and_merge_without_lost_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);children=[];original=subprocess.Popen
            def start(command,**kwargs):
                receipt=command[-1]
                code=('import json,time,pathlib; p=pathlib.Path('+repr(receipt)+'); '
                      's=json.loads(p.read_text()); s["started"]=time.time(); '
                      'p.write_text(json.dumps(s)); time.sleep(.5); '
                      's["finished"]=time.time(); s["worker_phase"]="done"; '
                      's["jobs"][0]["state"]="complete"; p.write_text(json.dumps(s))')
                child=original([sys.executable,'-c',code],**kwargs);children.append(child);return child
            with patch.object(q,'R',root),patch.object(q.subprocess,'Popen',side_effect=start),patch.object(q.p,'save'):
                q.launch(self.state,'train','train',self.state['jobs'][1])
                q.launch(self.state,'eval','eval',self.state['jobs'][0])
                for child in children:self.assertEqual(child.wait(timeout=10),0)
                data=[q.p.read(Path(w['path'])) for w in self.state['parallel_workers'].values()]
                self.assertLess(max(d['started'] for d in data),min(d['finished'] for d in data))
                self.assertIsNone(q.collect(self.state,'train'));self.assertIsNone(q.collect(self.state,'eval'))
            self.assertTrue(all(j['state']=='complete' for j in self.state['jobs']))
            self.assertFalse(self.state['parallel_workers'])


if __name__=='__main__':unittest.main()
