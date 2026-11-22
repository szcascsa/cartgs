from common import *
from baseline import previous_runtime
from deployment_runtime import DeploymentRenderer
from large_scene import LARGE
import subprocess


def snapshot():
    def query(args):return subprocess.check_output(['nvidia-smi',*args,'--format=csv,noheader'],text=True).strip()
    return dict(gpu=query(['--query-gpu=utilization.gpu,memory.used,clocks.sm,power.draw,temperature.gpu']),
        processes=query(['--query-compute-apps=pid,used_memory']))


def distribution(values):
    return dict(median=float(np.median(values)),mean=float(np.mean(values)),
        p90=float(np.percentile(values,90)),p99=float(np.percentile(values,99)))


@torch.no_grad()
def main():
    repeat=int(sys.argv[1]) if len(sys.argv)>1 else 1
    profile='--profile' in sys.argv;ablation='--ablation' in sys.argv
    results=[]
    for group in ['tum','replica']:
        s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
        engines={'v27':previous_runtime(s),'optimized':DeploymentRenderer(s.model,s.gaussians,s.config)}
        if ablation:
            for name,prefix,visible,small in [('prefix',True,False,False),('visible',False,True,False),
                    ('small_tile',False,False,True),('no_prefix',False,True,True)]:
                engines[name]=DeploymentRenderer(s.model,s.gaussians,s.config,
                    selector_static_prefix=prefix,visible_view=visible,small_compensation_tile=small)
        conditions=[('1pct',dict(ratio_budget=.01)),('128',dict(budget=128)),('1000',dict(budget=1000)),
            ('2000',dict(budget=2000)),('5000',dict(budget=5000)),('10000',dict(budget=10000)),
            ('10pct',dict(ratio_budget=.1)),('70pct',dict(ratio_budget=.7)),('mixed',None)]
        if profile:conditions=[row for row in conditions if row[0] in ['1000','5000','70pct']]
        if ablation:conditions=[row for row in conditions if row[0] in ['1pct','128','1000','5000','70pct']]
        # Warm both runtimes for two seconds, including the first small-ratio plan.
        warm_until=time.perf_counter()+2.;i=0
        warm_requests=[dict(budget=1000),dict(budget=5000),dict(ratio_budget=.01),dict(ratio_budget=.7)]
        while time.perf_counter()<warm_until:
            cam=camera(s,(i*73)%len(s.views))
            for engine in engines.values():
                packet,_,_=engine.prepare(cam,**warm_requests[i%4])
                render_prepared(cam,packet,s.background)
            torch.cuda.synchronize();i+=1
        for condition,fixed in conditions:
            samples={name:dict(prepare_ms=[],total_ms=[],native_ms=[]) for name in engines}
            actual=[];before=snapshot()
            for i in range(36 if profile else 150):
                cam=camera(s,(i*73+19)%len(s.views))
                request=fixed or ([dict(budget=k) for k in [1,128,1000,2000,5000,10000]]+
                    [dict(ratio_budget=r) for r in [.01,.08,.3,.7]])[i%10]
                names=list(engines);shift=(i+repeat)%len(names);names=names[shift:]+names[:shift]
                if ((i+repeat)//len(names))%2:names.reverse()
                for name in names:
                    torch.cuda.synchronize()
                    if profile and i>=20:torch.cuda.nvtx.range_push(f'prepare/{name}/{group}/{condition}')
                    started=time.perf_counter()
                    packet,ids,stats=engines[name].prepare(cam,**request)
                    if profile and i>=20:torch.cuda.nvtx.range_pop()
                    prepared=time.perf_counter()
                    if not profile and not ablation:render_prepared(cam,packet,s.background);torch.cuda.synchronize()
                    finished=time.perf_counter()
                    if i>=20:
                        samples[name]['prepare_ms'].append((prepared-started)*1000)
                        samples[name]['total_ms'].append((finished-started)*1000)
                        samples[name]['native_ms'].append(stats['prepare_ms'])
                        if name=='optimized':actual.append(dict(N=stats['candidate_gaussians'],K=len(ids)))
            results.append(dict(group=group,condition=condition,actual=actual,gpu_before=before,gpu_after=snapshot(),
                summary={n:{k:distribution(v) for k,v in values.items()} for n,values in samples.items()},samples=samples))
            kind='PROFILE_SAMPLES' if profile else ('ABLATION' if ablation else 'BENCHMARK')
            save(ROOT/'reports'/f'{kind}_{repeat}.json',dict(repeat=repeat,results=results))
            print(group,condition,{n:v['prepare_ms']['median'] for n,v in results[-1]['summary'].items()},flush=True)
        del engines,s;torch.cuda.empty_cache()


if __name__=='__main__':main()
