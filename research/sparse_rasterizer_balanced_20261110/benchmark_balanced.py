"""同输入逐项配对；保留V1真实二进制，区分正式空闲和受干扰结果。"""
from benchmark import *
import importlib.util


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--all-scenes',action='store_true')
    parser.add_argument('--mode',type=int,default=4)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    old_path=Path('/home/prmi/zwj/experiments/sparse_rasterizer_20261109/runtime.py')
    spec=importlib.util.spec_from_file_location('sparse_original_runtime',old_path)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    variants={'bitset_v1':old.SparseRenderer(2),'warp_v1':old.SparseRenderer(1)}
    if args.quick:
        variants.update(hierarchical=SparseRenderer(3),compact=SparseRenderer(4))
    else:variants['balanced']=SparseRenderer(args.mode)
    selected=['Replica__office1__mono','Replica__office2__rgbd','Replica__room0__rgbd','Replica__room2__mono']
    if args.all_scenes:selected=[s['slug'] for s in SCENES]
    rows=[];workloads=[];loads=[]
    conditions=[('local1pct',dict(ratio_budget=.01)),('local5pct',dict(ratio_budget=.05)),('local10pct',dict(ratio_budget=.1)),
        ('local70pct',dict(ratio_budget=.7)),('local1000',dict(budget=1000)),('local5000',dict(budget=5000)),('full',{})]
    if args.quick:conditions=[c for c in conditions if c[0] in ['local1pct','local10pct','full']]
    for slug in selected:
        scene,source=load_scene(slug)
        frames=[(19+137*i)%len(source.scene.views) for i in range(4 if args.quick else 8)]
        for condition,request in conditions:
            load=external_load();loads.append(load);assert not load['active'],load
            packets=[]
            for frame in frames:
                cam=source.camera(frame)
                if condition=='full':packet,_=source.runtime.full()
                else:packet,_,_=source.runtime.prepare(cam,**request)
                original=render_prepared(cam,packet,source.scene.background)['render']
                for engine in variants.values():
                    image,_=engine.render(cam,packet,source.scene.background)
                    assert torch.equal(image.contiguous().view(torch.uint8),original.contiguous().view(torch.uint8)),(slug,frame,condition)
                packets.append((frame,cam,packet))
                workloads.append(dict(scene=scene['id'],condition=condition,frame=frame,K=len(packet['inputs']['means3D'])))
            functions={'original':lambda cam,p:render_prepared(cam,p,source.scene.background)['render']}
            functions.update({n:(lambda cam,p,e=e:e.render(cam,p,source.scene.background)[0]) for n,e in variants.items()})
            for _,cam,packet in packets:
                for fn in functions.values():
                    for _ in range(3):image=fn(cam,packet)
            for repeat in range(args.repeats):
                for i in range(24 if args.quick else 40):
                    frame,cam,packet=packets[i%len(packets)];names=list(functions)
                    shift=(i+repeat)%len(names);names=names[shift:]+names[:shift]
                    if (i+repeat)%2:names.reverse()
                    for name in names:
                        torch.cuda.synchronize();start=time.perf_counter()
                        image=functions[name](cam,packet);torch.cuda.synchronize()
                        rows.append(dict(scene=scene['id'],condition=condition,K=len(packet['inputs']['means3D']),frame=frame,
                            repeat=repeat,iteration=i,variant=name,wall_ms=(time.perf_counter()-start)*1000))
            write_csv(args.output/'samples.csv',rows)
            load=external_load();loads.append(load);(args.output/'LOAD.json').write_text(json.dumps(loads,indent=2))
            assert not load['active'],load
            print('BALANCED_BENCHMARK',slug,condition,flush=True)
        del source,packet,packets
        torch.cuda.empty_cache()
    results=[]
    for scene,condition in sorted(set((r['scene'],r['condition']) for r in rows)):
        group=[r for r in rows if (r['scene'],r['condition'])==(scene,condition)]
        times={name:st.median(r['wall_ms'] for r in group if r['variant']==name) for name in functions}
        for name,ms in times.items():
            results.append(dict(scene=scene,condition=condition,variant=name,ms=ms,speedup_original=times['original']/ms,
                speedup_v1=times['bitset_v1']/ms,
                repeat_speedups=[st.median(r['wall_ms'] for r in group if r['variant']=='original' and r['repeat']==j)/
                    st.median(r['wall_ms'] for r in group if r['variant']==name and r['repeat']==j) for j in range(args.repeats)]))
    write_csv(args.output/'workloads.csv',workloads)
    result=dict(passed=True,formal_idle_benchmark=True,quick=args.quick,all_scenes=args.all_scenes,repeats=args.repeats,
        results=results,rows=len(rows),workspace_bytes={n:e.native.workspace_bytes() for n,e in variants.items()},
        pixel_bitwise=True,observations=loads)
    (args.output/'SUMMARY.json').write_text(json.dumps(result,indent=2))
    print('BALANCED_BENCHMARK_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
