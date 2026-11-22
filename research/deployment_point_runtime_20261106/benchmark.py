from common import *
from deployment_runtime import DeploymentRenderer
import statistics
import subprocess


def describe(v):
    return dict(mean_ms=statistics.mean(v),median_ms=statistics.median(v),p90_ms=float(np.percentile(v,90)),
                p99_ms=float(np.percentile(v,99)),under_2ms_fraction=statistics.mean(x<2 for x in v))


@torch.no_grad()
def main():
    s=scene();repeat=int(sys.argv[1]) if len(sys.argv)>1 else 1
    engines={'v3':reference_renderer(s),'deployment':DeploymentRenderer(s.model,s.gaussians,s.config)}
    results=[]
    for case in ['r01','r70','changing_ratio','changing_points']:
        times={k:{stage:[] for stage in ['prepare','total']} for k in engines};ks=[]
        for i in range(812):
            frame=(i-20)%792;cam=camera(s,frame)
            request=dict(ratio_budget=.01 if case=='r01' else .7)
            if case=='changing_ratio':request=dict(ratio_budget=RATIOS[frame%len(RATIOS)])
            if case=='changing_points':request=dict(budget=[1000,2000,3000,4000,5000,10000][frame%6])
            order=list(engines)
            if (i+repeat)%2:order.reverse()
            for tag in order:
                kwargs=dict(request,view_conditioned=True)
                if tag=='v3':kwargs['measure_time']=False
                torch.cuda.synchronize();t=time.perf_counter()
                data,ids,stats=engines[tag].prepare(cam,**kwargs)
                # prepare contains its own finite-value completion dependency.
                p=time.perf_counter();render_prepared(cam,data,s.background);torch.cuda.synchronize();end=time.perf_counter()
                if i>=20:
                    times[tag]['prepare'].append((p-t)*1000);times[tag]['total'].append((end-t)*1000)
            if i>=20:ks.append(len(ids))
        row=dict(case=case,frames=792,K_min=min(ks),K_max=max(ks),unique_K=len(set(ks)),
                 summary={tag:{stage:describe(v) for stage,v in stages.items()} for tag,stages in times.items()},raw_ms=times)
        results.append(row);save(ROOT/'reports'/f'BENCHMARK_{repeat}.json',dict(results=results,graph=False,dynamic_camera=True))
        print(json.dumps({k:v for k,v in row.items() if k!='raw_ms'}),flush=True)
    save(ROOT/'reports'/f'GPU_STATE_{repeat}.json',dict(processes=subprocess.check_output(
        ['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv'],text=True)))


if __name__=='__main__':main()
