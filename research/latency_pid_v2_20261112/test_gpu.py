"""V2异步事件、积压反馈、重置、单Stream和固定K等价验收。"""
from pathlib import Path
import importlib.util
import json
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
from latency_pid import pid_extension,LatencyPIDRenderer
import torch


def main():
    module=pid_extension(str(ROOT/'build'))
    config=dict(target_ms=.2)
    passed=[]
    with torch.no_grad():
        p=module.AsyncPID(config,0,8)
        assert p.request([],False,1.)==3000
        token,a,z=p.start_native(3000,10000)
        # 独立测试中原生CUDA Event句柄通过Torch同Stream记录。
        # 真正render-timed路径在benchmark使用Mode40与原版逐位比较。
        p.finish(token,False);torch.cuda.synchronize();p.poll()
        assert p.stats()['stale_samples']==1;passed.append('cancel_native')
        for j in range(4):
            p.request([],False,2.+j);t=p.start(3000,10000)
            torch.cuda._sleep(100000);p.finish(t,True)
        torch.cuda.synchronize();p.poll()
        assert p.stats()['feedback_samples']==4 and p.stats()['dropped_samples']==0
        passed.append('async_receipts')
        p.request([],True,10.);assert p.stats()['epoch']==1;passed.append('reset')
        p.set_target(.3);assert p.stats()['target_ms']==.3;passed.append('target')
        p.request([],False,11.);t=p.start(0,0);p.finish(t,True)
        torch.cuda.synchronize();p.poll();passed.append('empty')
        other=torch.cuda.Stream()
        rejected=False
        try:
            with torch.cuda.stream(other):p.request([],False,12.)
        except RuntimeError:rejected=True
        assert rejected;passed.append('stream_guard')
        p.close();p.close();passed.append('idempotent_close')
        rejected=False
        try:p.request([],False,13.)
        except RuntimeError:rejected=True
        assert rejected;passed.append('closed_guard')
        p=module.AsyncPID(config,0,1)
        p.request([],False,1.);t=p.start(3000,10000);torch.cuda._sleep(10000000);p.finish(t,True)
        p.request([],False,2.);t=p.start(3000,10000);p.finish(t,True)
        assert p.stats()['dropped_samples']==1
        torch.cuda.synchronize();p.poll();p.close();passed.append('bounded_ring')
        # 必须在同一poll中遇到积压样本，控制只消费最新一条，且收据不丢。
        p=module.AsyncPID(config,0,8)
        p.request([],False,1.);t=p.start(3000,10000);torch.cuda._sleep(20000000);p.finish(t,True)
        for j in range(3):
            p.request([],False,2.+j);t=p.start(3000,10000);p.finish(t,True)
        torch.cuda.synchronize();p.poll();s=p.stats()
        assert s['feedback_samples']==4 and s['superseded_samples']>=1
        p.close();passed.append('coalesce_stale_feedback')
    result=dict(passed=passed,count=len(passed),binary=module.__file__)
    (ROOT/'reports/GPU_TESTS.json').write_text(json.dumps(result,indent=2))
    print('PID_V2_GPU_PASS',json.dumps(result),flush=True)


if __name__=='__main__':main()
