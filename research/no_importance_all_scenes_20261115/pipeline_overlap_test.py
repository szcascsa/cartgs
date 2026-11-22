"""仅CPU调度测试，不读取或修改正式实验状态。"""
import collections
import threading
import unittest
from unittest.mock import patch
import run


class OverlapTest(unittest.TestCase):
    def test_two_training_and_two_evaluations_overlap(self):
        jobs=[dict(id=str(i),state='trained' if i<2 else 'pending',evaluations=[]) for i in range(5)]
        release=threading.Event();condition=threading.Condition();active=collections.Counter();trained=[];evaluated=[];errors=[]
        def enter(role):
            with condition:active[role]+=1;condition.notify_all()
            if not release.wait(5):raise RuntimeError('两类任务未并行启动')
        def train(job,*args):
            enter('train');trained.append(job['id']);job['state']='trained'
            with condition:active['train']-=1
            return 'fake_model'
        def evaluate(job,*args):
            assert job['state']=='trained','禁止在模型训练结束前评价'
            enter('eval');evaluated.append(job['id']);job['state']='complete'
            with condition:active['eval']-=1
        def main():
            try:run.pipelined_work(jobs,dict(jobs=jobs))
            except BaseException as error:errors.append(error)
        with patch.object(run,'train',train),patch.object(run,'evaluate_job',evaluate),patch.object(run,'save',lambda _:None):
            thread=threading.Thread(target=main);thread.start()
            try:
                with condition:self.assertTrue(condition.wait_for(lambda:active['train']==2 and active['eval']==2,5))
            finally:release.set();thread.join(10)
        self.assertFalse(thread.is_alive());self.assertEqual(errors,[])
        self.assertCountEqual(trained,['2','3','4']);self.assertCountEqual(evaluated,[str(i) for i in range(5)])
        self.assertTrue(all(j['state']=='complete' for j in jobs))

    def test_skipped_training_does_not_evaluate(self):
        jobs=[dict(id='oom',state='pending',evaluations=[])]
        def train(job,*args):job['state']='skipped_oom';return None
        with patch.object(run,'train',train),patch.object(run,'save',lambda _:None),patch.object(run,'evaluate_job') as evaluate:
            run.pipelined_work(jobs,dict(jobs=jobs));evaluate.assert_not_called()
        self.assertEqual(jobs[0]['state'],'skipped_oom')

    def test_failed_evaluation_keeps_other_jobs_running(self):
        jobs=[dict(id=str(i),state='trained',evaluations=[]) for i in range(3)]
        def evaluate(job,*args):
            if job['id']=='1':raise ValueError('test_failure')
            job['state']='complete'
        with patch.object(run,'evaluate_job',evaluate),patch.object(run,'save',lambda _:None):
            run.pipelined_work(jobs,dict(jobs=jobs))
        self.assertEqual([j['state'] for j in jobs],['complete','failed','complete'])


if __name__=='__main__':unittest.main()
