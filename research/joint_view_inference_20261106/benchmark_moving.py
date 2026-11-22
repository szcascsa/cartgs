"""真实792帧相机逐帧变化；包括Graph首次捕获/淘汰，避免只报热态收益。"""
from common import *
import statistics
import subprocess


@torch.no_grad()
def main():
    s=scene();results=[]
    for condition,request in [('ratio_0.01',dict(ratio_budget=.01)),('ratio_0.1',dict(ratio_budget=.1)),
                              ('ratio_0.7',dict(ratio_budget=.7)),('points_2000',dict(budget=2000)),('points_5000',dict(budget=5000))]:
        engines={'pytorch':reference_renderer(s),'cuda':FixedBudgetRenderer(s.model,s.gaussians,s.config,0,'cuda'),
                 'cuda_graph':FixedBudgetRenderer(s.model,s.gaussians,s.config,0,'cuda_graph')}
        captures=[];original=engines['cuda_graph'].inference._capture
        def capture(*args,**kwargs):captures.append(len(args[0]));return original(*args,**kwargs)
        engines['cuda_graph'].inference._capture=capture
        samples={n:[] for n in engines};sizes=[]
        for frame in range(792):
            cam=camera(s,frame);order=list(engines)
            if frame%2:order.reverse()
            for name in order:
                kwargs=dict(request,view_conditioned=True)
                if name!='pytorch':kwargs['measure_time']=False
                torch.cuda.synchronize();t=time.perf_counter()
                data,ids,_=engines[name].prepare(cam,**kwargs);render_prepared(cam,data,s.background);torch.cuda.synchronize()
                samples[name].append((time.perf_counter()-t)*1000)
            sizes.append(len(ids))
        row=dict(condition=condition,frames=792,unique_K=len(set(sizes)),min_K=min(sizes),max_K=max(sizes),captures=len(captures),
            total_ms={n:sum(v) for n,v in samples.items()},median_ms={n:statistics.median(v) for n,v in samples.items()},
            p90_ms={n:float(np.percentile(v,90)) for n,v in samples.items()},p99_ms={n:float(np.percentile(v,99)) for n,v in samples.items()},raw_ms=samples)
        results.append(row);save(ROOT/'reports/MOVING_BENCHMARK.json',dict(results=results,includes_graph_capture=True,
            gpu_activity=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True)))
        print(json.dumps({k:v for k,v in row.items() if k!='raw_ms'}),flush=True)
        del engines;torch.cuda.empty_cache()


if __name__=='__main__':main()
