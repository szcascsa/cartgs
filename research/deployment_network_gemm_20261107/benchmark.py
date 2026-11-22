"""只测网络与准备，不把Rasterizer计入加速结果。"""
from common import *
from baseline import previous_runtime
from deployment_runtime import DeploymentRenderer
from large_scene import LARGE
import statistics
import subprocess


def stats(x):
    return dict(mean_ms=statistics.mean(x),median_ms=statistics.median(x),
        p90_ms=float(np.percentile(x,90)),p99_ms=float(np.percentile(x,99)))


@torch.no_grad()
def main():
    run=int(sys.argv[1]);group=sys.argv[2];frames=792 if group=='tum' else 411
    s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
    engines={'v11':previous_runtime(s),'optimized':DeploymentRenderer(s.model,s.gaussians,s.config)};result=[]
    for case in ['r01','r10','r70','changing_ratio','changing_points','jumping_view']:
        values={t:[] for t in engines};ks=[];ns=[]
        for i in range(frames+20):
            frame=(i-20)%len(s.views);cam=camera(s,frame)
            request=dict(ratio_budget={'r01':.01,'r10':.1}.get(case,.7))
            if case=='changing_ratio':request=dict(ratio_budget=RATIOS[frame%len(RATIOS)])
            if case=='changing_points':request=dict(budget=[1000,2000,3000,4000,5000,10000][frame%6])
            if case=='jumping_view':
                cam=camera(s,(frame*73+19)%len(s.views))
                request=dict(ratio_budget=RATIOS[frame%len(RATIOS)]) if frame%2 else dict(budget=[1000,5000,10000][frame%3])
            order=list(engines)
            if (i+run)%2:order.reverse()
            for tag in order:
                torch.cuda.synchronize();start=time.perf_counter()
                data,ids,out=engines[tag].prepare(cam,**request)
                torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
                if i>=20:values[tag].append(elapsed)
            if i>=20:ks.append(len(ids));ns.append(out['candidate_gaussians'])
        result.append(dict(case=case,frames=frames,N_min=min(ns),N_max=max(ns),K_min=min(ks),K_max=max(ks),
            summary={t:stats(v) for t,v in values.items()},raw_ms=values))
        save(ROOT/'reports'/f'BENCHMARK_{group}_{run}.json',dict(results=result,baseline='deployment_v11',
            rasterizer_in_timed_region=False,camera_or_ratio_result_cache=False))
        print(group,run,case,result[-1]['summary'],flush=True)
    save(ROOT/'reports'/f'GPU_STATE_{group}_{run}.json',dict(processes=subprocess.check_output(
        ['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv'],text=True)))


if __name__=='__main__':main()
