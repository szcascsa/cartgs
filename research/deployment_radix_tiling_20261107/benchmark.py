from common import *
from baseline import previous_runtime
from deployment_runtime import DeploymentRenderer
import statistics
import subprocess
from large_scene import LARGE


def stats(values):
    return dict(mean_ms=statistics.mean(values),median_ms=statistics.median(values),
        p90_ms=float(np.percentile(values,90)),p99_ms=float(np.percentile(values,99)),
        under2ms_fraction=float(np.mean(np.array(values)<2)))


@torch.no_grad()
def main():
    group=sys.argv[3] if len(sys.argv)>3 else 'tum'
    s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
    run=int(sys.argv[1]);frames=int(sys.argv[2]) if len(sys.argv)>2 else len(s.views)
    engines={'v7':previous_runtime(s),'optimized':DeploymentRenderer(s.model,s.gaussians,s.config)}
    result=[]
    for case in ['r01','r70','changing_ratio','changing_points','jumping_view']:
        values={name:{stage:[] for stage in ['prepare','total']} for name in engines};ks=[]
        for i in range(frames+20):
            frame=(i-20)%len(s.views);cam=camera(s,frame)
            request=dict(ratio_budget=.01 if case=='r01' else .7)
            if case=='changing_ratio':request=dict(ratio_budget=RATIOS[frame%len(RATIOS)])
            if case=='changing_points':request=dict(budget=[1000,2000,3000,4000,5000,10000][frame%6])
            if case=='jumping_view':
                cam=camera(s,(frame*73+19)%len(s.views))
                request=dict(ratio_budget=RATIOS[frame%len(RATIOS)]) if frame%2 else dict(budget=[1000,5000,10000][frame%3])
            names=list(engines)
            if (i+run)%2:names.reverse()
            for name in names:
                torch.cuda.synchronize();start=time.perf_counter()
                data,ids,out=engines[name].prepare(cam,**request)
                prepared=time.perf_counter();render_prepared(cam,data,s.background);torch.cuda.synchronize();end=time.perf_counter()
                if i>=20:values[name]['prepare'].append((prepared-start)*1000);values[name]['total'].append((end-start)*1000)
            if i>=20:ks.append(len(ids))
        row=dict(case=case,frames=frames,K_min=min(ks),K_max=max(ks),unique_K=len(set(ks)),
            summary={tag:{stage:stats(v) for stage,v in stages.items()} for tag,stages in values.items()},raw_ms=values)
        result.append(row);save(ROOT/'reports'/f'BENCHMARK_{group}_{run}.json',dict(results=result,frames_per_case=frames,
            compared_to='deployment_v7',group=group,graph=False,mask_cache=False,ratio_feature_cache=False))
        print(json.dumps({k:v for k,v in row.items() if k!='raw_ms'}),flush=True)
    save(ROOT/'reports'/f'GPU_STATE_{run}.json',dict(processes=subprocess.check_output(
        ['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv'],text=True)))


if __name__=='__main__':main()
