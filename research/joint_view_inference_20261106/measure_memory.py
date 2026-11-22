"""单后端独立进程测显存和ATen分配，Profiler时间不计入性能结果。"""
from common import *
import argparse


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('backend',choices=['pytorch','cuda','cuda_graph']);args=parser.parse_args()
    s=scene();torch.cuda.synchronize();base=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved()
    e=reference_renderer(s) if args.backend=='pytorch' else FixedBudgetRenderer(s.model,s.gaussians,s.config,0,args.backend)
    cam=camera(s,396)
    for b in [1000,2000,3000,4000,5000,6000]:
        for _ in range(4):data,_,_=e.prepare(cam,b,view_conditioned=True);render_prepared(cam,data,s.background)
    del data;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    # CUDA Graph replay + CUPTI hangs in this installed profiler's postprocessing.
    # Use identical CPU allocator instrumentation for all backends; Nsight records GPU work separately.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],profile_memory=True,record_shapes=True) as prof:
        for _ in range(10):
            data,_,_=e.prepare(cam,2000,view_conditioned=True);render_prepared(cam,data,s.background)
    torch.cuda.synchronize()
    ops=[]
    for event in prof.key_averages():
        ops.append(dict(name=event.key,count=event.count,cuda_memory=getattr(event,'self_cuda_memory_usage',0),cpu_memory=event.self_cpu_memory_usage))
    engine=e.inference
    save(ROOT/'reports'/('MEMORY_'+args.backend+'.json'),dict(backend=args.backend,baseline_allocated=base,baseline_reserved=reserved,
        allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),
        additional_retained_bytes=torch.cuda.memory_allocated()-base,graphs=0 if engine is None else len(engine.graphs),
        graph_accounted_bytes=0 if engine is None else engine.graph_bytes,operations=ops,profiled_iterations=10,
        profiler_activities=['CPU'],cuda_allocation_state_from_torch=True,
        scope='每后端独立进程，相同场景与预热；额外量包含当前输出和缓存，不把reserved全部归于新增活跃显存'))


if __name__=='__main__':main()
