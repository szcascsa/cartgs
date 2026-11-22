"""独立于正式墙钟计时采集CUDA Event和Kineto Kernel/同步证据。"""
from benchmark import *
import importlib.util


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('old_profile_runtime',Path('/home/prmi/zwj/experiments/sparse_rasterizer_20261109/runtime.py'))
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    engines={'bitset_v1':old.SparseRenderer(2),'warp_v1':old.SparseRenderer(1),'balanced':SparseRenderer(4)}
    scene,source=load_scene('Replica__office2__rgbd');cam=source.camera(293)
    rows=[];events=[]
    for condition,ratio in [('local1pct',.01),('local10pct',.1),('local70pct',.7),('full',None)]:
        if ratio:packet,_,_=source.runtime.prepare(cam,ratio_budget=ratio)
        else:packet,_=source.runtime.full()
        functions={'original':lambda:render_prepared(cam,packet,source.scene.background)['render']}
        functions.update({name:(lambda e=engine:e.render(cam,packet,source.scene.background)[0]) for name,engine in engines.items()})
        for name,fn in functions.items():
            for _ in range(5):image=fn()
            torch.cuda.synchronize()
            stream_samples=[]
            for _ in range(40):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record();image=fn();end.record();end.synchronize()
                stream_samples.append(start.elapsed_time(end))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(6):
                    with torch.profiler.record_function(f'{condition}/{name}'):image=fn()
                torch.cuda.synchronize()
            path=args.output/f'trace_{condition}_{name}.json';prof.export_chrome_trace(str(path))
            trace=json.loads(path.read_text())['traceEvents']
            kernels=[e for e in trace if e.get('cat')=='kernel']
            sync=[e for e in trace if e.get('cat')=='cuda_runtime' and 'Synchronize' in e.get('name','')]
            details={}
            for e in kernels:
                value=details.setdefault(e['name'],dict(calls=0,us=0.));value['calls']+=1;value['us']+=e['dur']
            rows.append(dict(condition=condition,variant=name,K=len(packet['inputs']['means3D']),
                cuda_event_ms=st.median(stream_samples),cuda_event_samples_ms=stream_samples,
                gpu_kernel_us=sum(e['dur'] for e in kernels)/6,kernels_per_frame=len(kernels)/6,
                memcpy_per_frame=sum(e.get('cat')=='gpu_memcpy' for e in trace)/6,
                memset_per_frame=sum(e.get('cat')=='gpu_memset' for e in trace)/6,
                sync_calls_per_frame=len(sync)/6,sync_cpu_us_per_frame=sum(e['dur'] for e in sync)/6,
                workspace_bytes=engines[name].native.workspace_bytes() if name in engines else None,
                kernel_details=details))
            print('PROFILE',condition,name,flush=True)
    (args.output/'SUMMARY.json').write_text(json.dumps(dict(passed=True,results=rows,
        timing_scope='CUDA Event为Stream区间，含Host发射空隙；Kineto只累计cat=kernel，独立于正式墙钟计时'),indent=2))


if __name__=='__main__':main()
