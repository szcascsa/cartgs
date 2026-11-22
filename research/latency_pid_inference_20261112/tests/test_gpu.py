"""GPU异步环、失效语义与单次推理调用检查；不是速度测量。"""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from latency_pid import LatencyPIDRenderer,pid_extension


class NativePIDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module=pid_extension(str(ROOT/'build'))

    def make(self,**options):
        return self.module.AsyncPID(options,0,1)

    def delayed(self,pid,pose=None):
        pid.request([] if pose is None else pose,False,0.)
        token=pid.start(3000,12000)
        torch.cuda._sleep(100_000_000)
        pid.finish(token,True)

    def test_config_validation_and_no_mutation(self):
        config={'target_ms':.5}
        pid=self.module.AsyncPID(config,0,2)
        self.assertEqual(config,{'target_ms':.5})
        self.assertEqual(pid.stats()['target_ms'],.5)
        pid.close()
        for config in [{'kp':float('nan')},{'target_ms':0},{'unknown':1},{'max_budget':0}]:
            with self.assertRaises(ValueError):self.module.AsyncPID(config,0,1)

    def test_nonblocking_ring_saturation(self):
        pid=self.make();self.delayed(pid)
        pid.request([],False,.01)
        self.assertEqual(pid.start(3000,12000),-1)
        self.assertEqual(pid.stats()['dropped_samples'],1)
        self.assertEqual(pid.stats()['pending_samples'],1)
        torch.cuda.synchronize();pid.poll()
        self.assertEqual(pid.stats()['updates'],1)
        self.assertEqual(pid.stats()['last_feedback_K'],3000)
        self.assertEqual(pid.stats()['last_feedback_frame'],1)
        pid.close()

    def test_target_discards_pending_feedback(self):
        pid=self.make();self.delayed(pid)
        pid.set_target(.8)
        torch.cuda.synchronize();pid.poll()
        self.assertEqual(pid.stats()['updates'],0)
        self.assertEqual(pid.stats()['stale_samples'],1)
        self.assertEqual(pid.stats()['target_ms'],.8)
        pid.close()

    def test_pose_cut_discards_pending_feedback(self):
        pid=self.make();pose=np.eye(4);self.delayed(pid,pose.ravel().tolist())
        pose[0,3]=1.
        pid.request(pose.ravel().tolist(),False,.01)
        torch.cuda.synchronize();pid.poll()
        self.assertEqual(pid.stats()['resets'],1)
        self.assertEqual(pid.stats()['updates'],0)
        self.assertEqual(pid.stats()['stale_samples'],1)
        pid.close()

    def test_stream_and_request_order(self):
        pid=self.make();pid.request([],False,0.)
        with self.assertRaises(RuntimeError):pid.request([],False,0.)
        with torch.cuda.stream(torch.cuda.Stream()):
            with self.assertRaises(RuntimeError):pid.request([],False,.01)
        pid.close()

    def test_zero_pool(self):
        pid=self.make();pid.request([],False,0.)
        token=pid.start(0,0);torch.cuda._sleep(100000);pid.finish(token,True)
        torch.cuda.synchronize();pid.poll()
        self.assertEqual(pid.stats()['updates'],0)
        self.assertEqual(pid.stats()['integral'],0.)
        pid.close()

    def test_wrapper_one_prepare_and_render(self):
        class Engine:
            calls=0
            def prepare(self,camera,budget):
                self.calls+=1
                k=min(budget,5)
                return {},torch.arange(k,device='cuda'),{'eligible_candidates':5}
        class Raster:
            calls=0
            fail=False
            def render(self,camera,packet,background):
                self.calls+=1
                if self.fail:raise ValueError('test render failure')
                return background[:,None,None].expand(3,2,2).clone(),torch.ones(5,device='cuda')
        class Camera:image_width=2;image_height=2
        engine,raster=Engine(),Raster()
        wrapped=LatencyPIDRenderer(engine,raster,torch.zeros(3,device='cuda'),build_directory=str(ROOT/'build'))
        with self.assertRaises(RuntimeError):wrapped.render(Camera())
        with torch.inference_mode():
            result=wrapped.render(Camera(),camera_to_world=np.eye(4),timestamp_s=0.)
            self.assertEqual(len(result['ids']),5)
            self.assertEqual(engine.calls,1);self.assertEqual(raster.calls,1)
            raster.fail=True
            with self.assertRaises(ValueError):wrapped.render(Camera(),timestamp_s=.01)
            torch.cuda.synchronize();status=wrapped.poll()
            self.assertEqual(status['stale_samples'],1)
            self.assertEqual(status['pending_samples'],0)
            with self.assertRaises(ValueError):wrapped.render(Camera(),camera_to_world=torch.eye(4,device='cuda'))
        wrapped.close()


if __name__=='__main__':unittest.main(verbosity=2)
