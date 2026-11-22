"""同输入交错两轮GPU Event/Host测量，独立记录热态与冷请求。"""
from common import *
import argparse
import csv
import statistics
import subprocess


def gpu_state():
    return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--round',type=int,default=1);parser.add_argument('--profile',action='store_true');args=parser.parse_args()
    s=scene();engines={'pytorch':reference_renderer(s)}
    engines.update({k:FixedBudgetRenderer(s.model,s.gaussians,s.config,0,'cuda_graph' if 'graph' in k else 'cuda')
                    for k in ['cuda','cuda_graph']})
    engines['cuda_no_view_fusion']=FixedBudgetRenderer(s.model,s.gaussians,s.config,0,'cuda')
    engines['cuda_no_view_fusion'].inference.fused_view=False
    engines['cuda_graph_no_timing']=FixedBudgetRenderer(s.model,s.gaussians,s.config,0,'cuda_graph')
    rows=[];cold=[];activity=[gpu_state()];start=time.monotonic()
    requests=[('ratio_0.01',dict(ratio_budget=.01)),('ratio_0.05',dict(ratio_budget=.05)),('ratio_0.1',dict(ratio_budget=.1)),
              ('ratio_0.3',dict(ratio_budget=.3)),('ratio_0.7',dict(ratio_budget=.7)),('points_2000',dict(budget=2000)),('points_5000',dict(budget=5000))]
    if args.profile:requests=[requests[0],requests[4],requests[5]]
    for frame in ([396] if args.profile else [0,198,396,594,791]):
        cam=camera(s,frame)
        for condition,request in requests:
            for name,e in engines.items():
                torch.cuda.synchronize();t=time.perf_counter();e.prepare(cam,view_conditioned=True,**request);torch.cuda.synchronize()
                cold.append(dict(frame=frame,condition=condition,backend=name,ms=(time.perf_counter()-t)*1000))
            iterations=12 if args.profile else 48
            for repeat in range(iterations):
                order=list(engines)
                if (repeat+args.round)%2:order.reverse()
                for name in order:
                    e=engines[name];kwargs=dict(request,view_conditioned=True)
                    if name!='pytorch':kwargs['measure_time']=name!='cuda_graph_no_timing'
                    torch.cuda.synchronize();t=time.perf_counter();begin=torch.cuda.Event(enable_timing=True);prepared=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    nvtx=f'inference/{name}/{condition}'
                    with torch.cuda.nvtx.range(nvtx):
                        begin.record();data,ids,stats=e.prepare(cam,**kwargs);prepared.record()
                        before_render=time.perf_counter();render_prepared(cam,data,s.background,return_radii=True);end.record();end.synchronize()
                    if repeat>=8:
                        rows.append(dict(round=args.round,frame=frame,condition=condition,backend=name,repeat=repeat,K=len(ids),
                            prepare_gpu_ms=begin.elapsed_time(prepared),raster_gpu_ms=prepared.elapsed_time(end),total_gpu_ms=begin.elapsed_time(end),
                            prepare_host_ms=(before_render-t)*1000,total_host_ms=(time.perf_counter()-t)*1000))
            activity.append(gpu_state());print('bench',args.round,frame,condition,'seconds',time.monotonic()-start,flush=True)
    summary=[]
    for condition,_ in requests:
        for name in engines:
            v=[r for r in rows if r['condition']==condition and r['backend']==name]
            item=dict(condition=condition,backend=name,samples=len(v),mean_K=statistics.mean(r['K'] for r in v))
            for key in ['prepare_gpu_ms','raster_gpu_ms','total_gpu_ms','prepare_host_ms','total_host_ms']:
                item[key]=statistics.median(r[key] for r in v);item[key+'_p90']=float(np.percentile([r[key] for r in v],90))
            summary.append(item)
    tag='PROFILE' if args.profile else 'BENCHMARK_'+str(args.round)
    with (ROOT/'reports'/(tag+'.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(ROOT/'reports'/(tag+'.json'),dict(summary=summary,cold=cold,gpu_activity=activity,seconds=time.monotonic()-start,
        graph_cache={k:dict(entries=len(e.inference.graphs),bytes=e.inference.graph_bytes) for k,e in engines.items() if e.inference is not None},
        allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated()))


if __name__=='__main__':main()
