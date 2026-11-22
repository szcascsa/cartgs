"""CPU准备分段实测；只在本进程包装记录函数，不修改被测源码。"""
import benchmark as b
from collections import defaultdict
import functools


@b.torch.no_grad()
def main():
    out=b.HERE/'reports/cpu_stage_profile01';out.mkdir(exist_ok=False)
    b.torch.set_num_threads(6);b.torch.set_num_interop_threads(1)
    scenes=b.json.loads((b.FROZEN/'reports/SCENES.json').read_text())
    scene=next(s for s in scenes if s['slug']=='Replica__office2__rgbd');cp=b.Path(scene['checkpoint'])
    model=b.load_selector(str(cp/'selector.pt'),'cpu').eval().requires_grad_(False)
    pc=b.GaussianModel(3);ply,=list((cp/'point_cloud').glob('iteration_*/point_cloud.ply'));pc.load_ply(str(ply),'cpu')
    runtime=b.FixedBudgetRenderer(model,pc,b.read_config(cp/'selector_config.yaml'),0,'pytorch')
    records=b.json.loads((cp/'cameras.json').read_text());cams=[b.camera(records[int(f*(len(records)-1))]) for f in [.19,.51,.83]]
    import anchor_budget,selector,renderer
    times=defaultdict(list);current=[None];profile=[False]
    def wrap(obj,name,label):
        original=getattr(obj,name)
        @functools.wraps(original)
        def timed(*args,**kwargs):
            if profile[0]:
                with b.torch.profiler.record_function(label):return original(*args,**kwargs)
            if current[0] is None:return original(*args,**kwargs)
            tick=b.time.perf_counter_ns();value=original(*args,**kwargs)
            times[(current[0],label)].append((b.time.perf_counter_ns()-tick)/1e6);return value
        setattr(obj,name,timed)
    for obj,name,label in [(runtime,'check_state','state_check'),(anchor_budget,'anchor_frustum_mask','frustum'),
        (anchor_budget,'compact_gaussians','compact_candidates'),(model,'joint_view_code','view_encoding'),
        (model,'forward_view_selection_features','selector_with_inputs'),(anchor_budget,'exact_budget_mask','strict_selection'),
        (selector,'compensated_attributes','compensation_with_decode'),(renderer,'prepare_render_data','final_packet')]:wrap(obj,name,label)
    for ratio in [.01,.1,.7]:
        for camera in cams:
            for _ in range(3):runtime.prepare(camera,ratio_budget=ratio,view_conditioned=True,measure_time=False)
        current[0]=ratio
        for iteration in range(30):
            tick=b.time.perf_counter_ns();runtime.prepare(cams[iteration%3],ratio_budget=ratio,view_conditioned=True,measure_time=False)
            times[(ratio,'total')].append((b.time.perf_counter_ns()-tick)/1e6)
        current[0]=None
    rows=[dict(ratio=ratio,stage=stage,mean_ms=sum(values)/len(values),median_ms=b.statistics.median(values),samples=len(values))
        for (ratio,stage),values in times.items()]
    b.csv_save(out/'STAGES.csv',rows)
    profile[0]=True
    with b.torch.profiler.profile(activities=[b.torch.profiler.ProfilerActivity.CPU],record_shapes=True) as trace:
        for ratio in [.01,.1,.7]:
            for camera in cams:runtime.prepare(camera,ratio_budget=ratio,view_conditioned=True,measure_time=False)
    (out/'ATEN_TABLE.txt').write_text(trace.key_averages().table(sort_by='self_cpu_time_total',row_limit=28))
    print('CPU_STAGE_SUMMARY',flush=True)
    for row in rows:print(row)
    print((out/'ATEN_TABLE.txt').read_text())
    assert not b.torch.cuda.is_initialized()
    b.save(out/'SUMMARY.json',dict(passed=True,threads=6,cuda_initialized=False,profiling_timing_not_used_for_latency=True,
        stage_timing='uninstrumented_ATen; light Python function timer wrappers; nested stages not summed',profile_calls=9,steady_calls=90))


if __name__=='__main__':main()
