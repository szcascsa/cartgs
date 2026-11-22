"""对当前常规/容量图候选进行独立算子归因，不用Profile时间替代正式配对结果。"""
from harness import *
from runtime import SparseRenderer
from graph_runtime import GraphRenderer
import gzip


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--nsight',action='store_true');args=parser.parse_args()
    output=HERE/'reports'/args.output;output.mkdir(exist_ok=False)
    rows=[]
    for slug in ['Replica__office2__rgbd','VECtor__corner-slow__mono']:
        scene,source=load_scene(slug);frame=views_for(source)[-2][1];camera=source.camera(frame)
        c=SparseRenderer(render_variant=5,capacity=256,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=2)
        g=GraphRenderer(plan_granularity=512);g.reserve(source.runtime.total,camera,source.scene.background)
        b=baseline_B()
        functions={'A':lambda p:render_prepared(camera,p,source.scene.background)['render'],
            'B':lambda p:b.render(camera,p,source.scene.background)[0],
            'C':lambda p:c.render(camera,p,source.scene.background)[0],
            'G':lambda p:g.render(camera,p,source.scene.background)[0]}
        packets=[]
        for condition in ['local0.01','local0.05','local0.1','local0.7','full']:
            packet,ids,stats=prepare(source,camera,frame,condition);packets.append((condition,packet,len(ids)))
            for function in functions.values():
                for _ in range(5):image=function(packet)
        torch.cuda.synchronize()
        if args.nsight:
            torch.cuda.cudart().cudaProfilerStart()
            for condition,packet,K in packets:
                for name,function in functions.items():
                    for _ in range(4):
                        torch.cuda.nvtx.range_push(slug+'/'+condition+'/'+name)
                        image=function(packet);torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
            break
        # 本机Torch 2.3.1 Kineto在容量Graph的Collection结束时阻塞；Graph改用Nsight。
        functions.pop('G')
        for condition,packet,K in packets:
            for name,function in functions.items():
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    for _ in range(8):
                        with torch.profiler.record_function(slug+'/'+condition+'/'+name):image=function(packet)
                    torch.cuda.synchronize()
                trace=output/(slug+'_'+condition+'_'+name+'.json');prof.export_chrome_trace(str(trace))
                events=json.loads(trace.read_text())['traceEvents'];details={}
                for event in events:
                    if event.get('cat')=='kernel':
                        value=details.setdefault(event['name'],dict(calls=0,us=0.))
                        value['calls']+=1;value['us']+=event['dur']
                runtime=[e for e in events if e.get('cat')=='cuda_runtime']
                samples=[]
                for _ in range(30):
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record();image=function(packet);end.record();end.synchronize();samples.append(start.elapsed_time(end))
                rows.append(dict(scene=scene['id'],condition=condition,frame=frame,variant=name,K=K,
                    cuda_event_ms=st.median(samples),kernel_us=sum(v['us'] for v in details.values())/8,
                    kernels=sum(v['calls'] for v in details.values())/8,
                    memcpy=sum(e.get('cat')=='gpu_memcpy' for e in events)/8,
                    memsets=sum(e.get('cat')=='gpu_memset' for e in events)/8,
                    synchronizes=sum('Synchronize' in e['name'] or 'Memcpy' in e['name'] for e in runtime)/8,
                    details=details))
                with gzip.open(str(trace)+'.gz','wb',compresslevel=6) as target:target.write(trace.read_bytes())
                trace.unlink()
            write_json(output/'SUMMARY.json',dict(passed=True,results=rows,nsight=False))
            print('CURRENT_PROFILE',scene['id'],condition,flush=True)
        del source,packets;torch.cuda.empty_cache()


if __name__=='__main__':main()
