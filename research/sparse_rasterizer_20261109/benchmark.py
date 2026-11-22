"""跳转视角、固定渲染输入、逐项消融与CUDA活动计时。"""
from pathlib import Path
import argparse
import csv
import json
import sys
import time
import subprocess
import statistics as st
HERE=Path(__file__).resolve().parent
PRIOR=Path('/home/prmi/zwj/experiments/compensator_distillation_replica_all_20261108/source/research/compensator_distillation_replica_all_20261108')
sys.path.insert(0,str(PRIOR))
from scene_runtime import *
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from runtime import SparseRenderer


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def external_load():
    raw=subprocess.check_output(['nvidia-smi','pmon','-c','1','-s','um'],text=True)
    active=[]
    for line in raw.splitlines():
        f=line.split()
        if len(f)>=7 and f[0].isdigit() and f[1].isdigit() and int(f[1])!=os.getpid():
            if any(v.isdigit() and int(v)>0 for v in f[3:7]):active.append(line)
    return dict(time=time.time(),raw=raw,active=active)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--allow-concurrent',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    assert (HERE/'reports/VALIDATION.json').exists(),'Pixel/Tile parity must pass first'
    variants={'control':SparseRenderer(0),'warp_expand':SparseRenderer(1),
        'short':SparseRenderer(0,True),'warp_short':SparseRenderer(1,True),'bitset':SparseRenderer(2)}
    observations=[external_load()]
    assert args.allow_concurrent or not observations[-1]['active'],observations[-1]
    samples=[];workloads=[];profiles=[]
    for slug in ['Replica__office1__mono','Replica__office2__rgbd','Replica__room0__rgbd','Replica__room2__mono']:
        scene,source=load_scene(slug)
        frames=[(19+137*i)%len(source.scene.views) for i in range(8)]
        for condition,ratio,budget in [('local1pct',.01,None),('local5pct',.05,None),('local10pct',.1,None),
                ('local70pct',.7,None),('local1000',None,1000),('local5000',None,5000),('full',None,None)]:
            packets=[]
            for frame in frames:
                cam=source.camera(frame)
                if condition=='full':packet,_=source.runtime.full()
                else:packet,_,_=source.runtime.prepare(cam,ratio_budget=ratio) if ratio else source.runtime.prepare(cam,budget=budget)
                original=render_prepared(cam,packet,source.scene.background)['render']
                for engine in variants.values():
                    image,_=engine.render(cam,packet,source.scene.background)
                    assert torch.equal(image.contiguous().view(torch.uint8),original.contiguous().view(torch.uint8)),(slug,frame,condition)
                # 单独采集负载，排除在计时之外。
                debug=variants['control'].render(cam,packet,source.scene.background,True)
                ranges=debug[8].cpu();lengths=ranges[:,1]-ranges[:,0]
                workloads.append(dict(scene=scene['id'],condition=condition,frame=frame,K=len(debug[1]),
                    tile_bbox_instances=int(debug[5].sum()),valid_tile_instances=int(lengths.sum()),tiles=len(lengths),
                    active_tiles=int((lengths>0).sum()),short_nonempty_tiles=int(((lengths>0)&(lengths<=32)).sum()),
                    list_p50=float(lengths.float().median()),list_p99=float(torch.quantile(lengths.float(),.99))))
                packets.append((frame,cam,packet))
            functions={'original':lambda cam,p:render_prepared(cam,p,source.scene.background)['render']}
            functions.update({n:(lambda cam,p,e=e:e.render(cam,p,source.scene.background)[0]) for n,e in variants.items()})
            # 相机/选点/Compensator只在准备输入时执行，不计入纯渲染计时。
            for _,cam,packet in packets:
                for fn in functions.values():
                    for _ in range(3):image=fn(cam,packet)
            torch.cuda.synchronize()
            for repeat in range(args.repeats):
                for i in range(40):
                    frame,cam,packet=packets[i%len(packets)]
                    names=list(functions)
                    if (i+repeat)%2:names.reverse()
                    else:names=names[(i+repeat)%len(names):]+names[:(i+repeat)%len(names)]
                    for name in names:
                        torch.cuda.synchronize();start=time.perf_counter()
                        image=functions[name](cam,packet);torch.cuda.synchronize()
                        samples.append(dict(scene=scene['id'],condition=condition,frame=frame,K=len(packet['inputs']['means3D']),
                            repeat=repeat,iteration=i,variant=name,wall_ms=(time.perf_counter()-start)*1000))
            observation=external_load();observations.append(observation)
            write_csv(args.output/'samples.csv',samples);write_csv(args.output/'workloads.csv',workloads)
            (args.output/'LOAD.json').write_text(json.dumps(observations,indent=2))
            assert args.allow_concurrent or not observation['active'],('External GPU activity',observation)
            if args.profile and slug=='Replica__office2__rgbd' and condition in ['local1pct','local70pct','full']:
                frame,cam,packet=packets[2]
                for name,fn in functions.items():
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA]) as prof:
                        for _ in range(10):
                            with torch.profiler.record_function(name):image=fn(cam,packet)
                        torch.cuda.synchronize()
                    trace_path=args.output/f'trace_{condition}_{name}.json'
                    prof.export_chrome_trace(str(trace_path))
                    events=json.loads(trace_path.read_text())['traceEvents']
                    # Kineto的GPU user annotation覆盖整段工作，不能重复计入Kernel。
                    kernels=[e for e in events if e.get('cat')=='kernel']
                    copies=[e for e in events if e.get('cat')=='gpu_memcpy']
                    memsets=[e for e in events if e.get('cat')=='gpu_memset']
                    duration=sum(e['dur'] for e in kernels)/10
                    profiles.append(dict(scene=scene['id'],condition=condition,variant=name,iterations=10,
                        gpu_us=duration,kernels_per_frame=len(kernels)/10,
                        copies_per_frame=len(copies)/10,memsets_per_frame=len(memsets)/10,
                        kernel_events=[dict(name=e['name'],us=e['dur']) for e in kernels]))
            print('BENCHMARK',slug,condition,'K',min(v['inputs']['means3D'].shape[0] for _,_,v in packets),
                max(v['inputs']['means3D'].shape[0] for _,_,v in packets),flush=True)
        del source,packets,packet
        torch.cuda.empty_cache()
    summary=[]
    for scene,condition in sorted({(r['scene'],r['condition']) for r in samples}):
        group=[r for r in samples if r['scene']==scene and r['condition']==condition]
        times={n:st.median(r['wall_ms'] for r in group if r['variant']==n) for n in functions}
        for name,ms in times.items():
            summary.append(dict(scene=scene,condition=condition,variant=name,wall_ms=ms,speedup=times['original']/ms,
                repeat_speedups=[st.median(r['wall_ms'] for r in group if r['variant']=='original' and r['repeat']==j)/
                    st.median(r['wall_ms'] for r in group if r['variant']==name and r['repeat']==j) for j in range(args.repeats)]))
    (args.output/'SUMMARY.json').write_text(json.dumps(dict(passed=True,repeats=args.repeats,results=summary,
        workspace_bytes={n:e.native.workspace_bytes() for n,e in variants.items()},
        selected_ids_and_render_inputs_identical=True,profiles=profiles,observations=observations,
        concurrent_diagnostic=args.allow_concurrent,formal_idle_benchmark=not args.allow_concurrent),indent=2))
    print('BENCHMARK_COMPLETE',len(samples),flush=True)


if __name__=='__main__':main()
