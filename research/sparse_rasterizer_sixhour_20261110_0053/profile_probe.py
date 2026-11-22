"""少量代表输入的真实Nsight/Kineto区间；不把Profiling计时当正式墙钟收益。"""
from harness import *
from runtime import SparseRenderer
import gzip


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--nsight',action='store_true');args=parser.parse_args()
    output=HERE/'reports'/args.output;output.mkdir(exist_ok=False)
    variants={'B':baseline_B(),'C1':SparseRenderer(render_variant=1),'C13':SparseRenderer(render_variant=1,fused_keys=True)}
    scene,source=load_scene('Replica__office2__rgbd');frame=views_for(source)[-2][1];camera=source.camera(frame)
    functions={'A':lambda packet:render_prepared(camera,packet,source.scene.background)['render']}
    functions.update({name:(lambda packet,e=engine:e.render(camera,packet,source.scene.background)[0]) for name,engine in variants.items()})
    rows=[]
    for condition in ['local0.01','local0.1','full']:
        packet,ids,stats=prepare(source,camera,frame,condition)
        for name,fn in functions.items():
            for _ in range(5):image=fn(packet)
        torch.cuda.synchronize()
        if args.nsight:
            torch.cuda.cudart().cudaProfilerStart()
            for name,fn in functions.items():
                for _ in range(5):
                    torch.cuda.nvtx.range_push(condition+'/'+name);image=fn(packet);torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
        else:
            for name,fn in functions.items():
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    for _ in range(6):
                        with torch.profiler.record_function(condition+'/'+name):image=fn(packet)
                    torch.cuda.synchronize()
                trace=output/(condition+'_'+name+'.json');prof.export_chrome_trace(str(trace))
                events=json.loads(trace.read_text())['traceEvents'];kernels=[e for e in events if e.get('cat')=='kernel']
                runtimes=[e for e in events if e.get('cat')=='cuda_runtime']
                details={}
                for event in kernels:
                    value=details.setdefault(event['name'],dict(calls=0,us=0.));value['calls']+=1;value['us']+=event['dur']
                rows.append(dict(scene=scene['id'],condition=condition,frame=frame,variant=name,K=len(ids),
                    kernels_per_frame=len(kernels)/6,kernel_us_per_frame=sum(e['dur'] for e in kernels)/6,
                    memsets_per_frame=sum(e.get('cat')=='gpu_memset' for e in events)/6,
                    memcpy_per_frame=sum(e.get('cat')=='gpu_memcpy' for e in events)/6,
                    synchronization_calls=sum('Synchronize' in e['name'] or 'Memcpy' in e['name'] for e in runtimes)/6,
                    runtime_us_per_frame=sum(e['dur'] for e in runtimes)/6,kernel_details=details))
                with gzip.open(str(trace)+'.gz','wb',compresslevel=6) as f:f.write(trace.read_bytes())
                trace.unlink()
        print('PROFILE_PROBE',condition,flush=True)
    write_json(output/'SUMMARY.json',dict(passed=True,results=rows,nsight=args.nsight))


if __name__=='__main__':main()
