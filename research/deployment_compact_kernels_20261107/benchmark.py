from common import *
from baseline import previous_runtime
from deployment_runtime import DeploymentRenderer
from large_scene import LARGE
import statistics
import csv
import subprocess


def gpu_snapshot():
    return subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,clocks.sm,power.draw,temperature.gpu',
        '--format=csv,noheader'],text=True).strip()


def distribution(values):
    return dict(median=float(np.median(values)),mean=float(np.mean(values)),
                p90=float(np.percentile(values,90)),p99=float(np.percentile(values,99)))


@torch.no_grad()
def main():
    repeat=int(sys.argv[1]) if len(sys.argv)>1 else 1
    profile='--profile' in sys.argv
    results=[]
    for group in ['tum','replica']:
        s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
        engines={'v21':previous_runtime(s),'optimized':DeploymentRenderer(s.model,s.gaussians,s.config)}
        conditions=[('1000',dict(budget=1000)),('2000',dict(budget=2000)),('5000',dict(budget=5000)),
                    ('10000',dict(budget=10000)),('1pct',dict(ratio_budget=.01)),
                    ('10pct',dict(ratio_budget=.1)),('70pct',dict(ratio_budget=.7)),('mixed',None)]
        if profile:conditions=[row for row in conditions if row[0] in ['1000','5000','70pct']]
        for condition,fixed in conditions:
            before_gpu=gpu_snapshot()
            samples={name:dict(prepare_ms=[],total_ms=[],event_ms=[]) for name in engines}
            actual=[]
            for i in range(40 if profile else 140):
                cam=camera(s,(i*73+19)%len(s.views))
                request=fixed or ([dict(budget=k) for k in [1,1000,2000,5000,10000]]+
                                  [dict(ratio_budget=r) for r in [.01,.08,.3,.7]])[i%9]
                names=list(engines)
                if (i+repeat)%2:names.reverse()
                for name in names:
                    torch.cuda.synchronize()
                    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    if profile and i>=20:torch.cuda.nvtx.range_push(f'prepare/{name}/{group}/{condition}')
                    start=time.perf_counter();begin.record()
                    packet,ids,stats=engines[name].prepare(cam,**request)
                    if profile and i>=20:torch.cuda.nvtx.range_pop()
                    end.record();end.synchronize()
                    prepared=time.perf_counter()
                    if not profile:render_prepared(cam,packet,s.background);torch.cuda.synchronize()
                    finished=time.perf_counter()
                    if i>=20:
                        samples[name]['prepare_ms'].append((prepared-start)*1000)
                        samples[name]['total_ms'].append((finished-start)*1000)
                        samples[name]['event_ms'].append(begin.elapsed_time(end))
                        if name=='optimized':actual.append(dict(N=stats['candidate_gaussians'],K=len(ids)))
            row=dict(group=group,condition=condition,request=fixed,actual=actual,gpu_before=before_gpu,gpu_after=gpu_snapshot(),
                     summary={n:{k:distribution(v) for k,v in x.items()} for n,x in samples.items()},samples=samples)
            results.append(row)
            name=f'BENCHMARK_{repeat}.json' if not profile else 'PROFILE_SAMPLES.json'
            save(ROOT/'reports'/name,dict(repeat=repeat,profiled=profile,results=results))
            print(group,condition,{n:x['prepare_ms']['median'] for n,x in row['summary'].items()},flush=True)
        del engines,s
        torch.cuda.empty_cache()


if __name__=='__main__':main()
