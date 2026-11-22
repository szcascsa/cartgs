"""调优集真实变视角、变K测试；冷建图与每帧重新准备分别记录。"""
from harness import *
from runtime import SparseRenderer
from graph_runtime import GraphRenderer


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--granularity',type=int,default=0)
    parser.add_argument('--compare-input-update',action='store_true');args=parser.parse_args()
    output=HERE/'reports'/args.output;output.mkdir(exist_ok=False)
    variants={'B':baseline_B(),'C':SparseRenderer(render_variant=5,capacity=256,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=2),
        'G':GraphRenderer(plan_granularity=args.granularity)};rows=[];cold=[];plans=[];checks=0
    if args.compare_input_update:variants['H']=GraphRenderer(plan_granularity=args.granularity,graph_input_update=True)
    for scene in SCENES:
        if scene['split']!='tune':continue
        item,source=load_scene(scene['slug']);frames=views_for(source)
        tick=time.perf_counter_ns();variants['G'].reserve(source.runtime.total,source.camera(frames[0][1]),source.scene.background)
        reserve_ms=(time.perf_counter_ns()-tick)/1e6
        if 'H' in variants:variants['H'].reserve(source.runtime.total,source.camera(frames[0][1]),source.scene.background)
        plans.append(dict(scene=scene['id'],stage='model_reserve',reserve_ms=reserve_ms,stats=variants['G'].native.stats()))
        functions={'A':lambda cam,p:render_prepared(cam,p,source.scene.background)['render']}
        functions.update({name:(lambda cam,p,e=engine:e.render(cam,p,source.scene.background)[0]) for name,engine in variants.items()})
        for condition in ['local0.01','local0.05','local0.1','local0.7','full']:
            observation=external_load();assert not observation['active'],observation
            packets=[]
            for motion,frame in frames:
                camera=source.camera(frame);packet,ids,stats=prepare(source,camera,frame,condition)
                ref=render_prepared(camera,packet,source.scene.background,return_radii=True)
                torch.cuda.synchronize();before=variants['G'].native.stats()
                tick=time.perf_counter_ns();image,radii=variants['G'].render(camera,packet,source.scene.background);torch.cuda.synchronize()
                elapsed=(time.perf_counter_ns()-tick)/1e6;after=variants['G'].native.stats()
                cold.append(dict(scene=scene['id'],condition=condition,frame=frame,motion=motion,K=len(ids),
                    graph_miss=after[1]>before[1],wall_ms=elapsed,workspace_bytes=after[4]))
                assert bits_equal(image,ref['render']) and bits_equal(radii,ref['radii']);checks+=1
                for name in ['B','C']+(['H'] if 'H' in variants else []):
                    image,radii=variants[name].render(camera,packet,source.scene.background)
                    assert bits_equal(image,ref['render']) and bits_equal(radii,ref['radii']);checks+=1
                packets.append((motion,frame,camera,packet))
            for repeat in range(4):
                for i,(motion,frame,camera,packet) in enumerate(packets):
                    names=list(functions);shift=(i+repeat)%len(names);names=names[shift:]+names[:shift]
                    if (i+repeat)%2:names.reverse()
                    for name in names:
                        before=variants['G'].native.stats() if name=='G' else None
                        torch.cuda.synchronize();tick=time.perf_counter_ns();image=functions[name](camera,packet);torch.cuda.synchronize()
                        elapsed=(time.perf_counter_ns()-tick)/1e6;after=variants['G'].native.stats() if name=='G' else None
                        rows.append(dict(scene=scene['id'],condition=condition,motion=motion,frame=frame,K=len(packet['inputs']['means3D']),
                            repeat=repeat,variant=name,scope='raster',wall_ms=elapsed,plan_miss=bool(after and after[1]>before[1])))
            for repeat in range(3):
                for i,(motion,frame,camera,_) in enumerate(packets):
                    names=list(functions);shift=(i+repeat)%len(names);names=names[shift:]+names[:shift]
                    if (i+repeat)%2:names.reverse()
                    paired=[]
                    for name in names:
                        before=variants['G'].native.stats() if name=='G' else None
                        torch.cuda.synchronize();tick=time.perf_counter_ns()
                        packet,ids,stats=prepare(source,camera,frame,condition);image=functions[name](camera,packet);torch.cuda.synchronize()
                        elapsed=(time.perf_counter_ns()-tick)/1e6;after=variants['G'].native.stats() if name=='G' else None
                        rows.append(dict(scene=scene['id'],condition=condition,motion=motion,frame=frame,K=len(ids),
                            repeat=repeat,variant=name,scope='pipeline',wall_ms=elapsed,plan_miss=bool(after and after[1]>before[1])))
                        paired.append((ids.clone(),image))
                    assert all(torch.equal(value[0],paired[0][0]) and bits_equal(value[1],paired[0][1]) for value in paired[1:]);checks+=1
            write_csv(output/'samples.csv',rows);write_csv(output/'cold_samples.csv',cold)
            plans.append(dict(scene=scene['id'],condition=condition,stats=variants['G'].native.stats()))
            write_json(output/'PLANS.json',plans);observation=external_load();assert not observation['active'],observation
            print('GRAPH_BENCHMARK',scene['id'],condition,flush=True)
        del source,packets,packet;torch.cuda.empty_cache()
    summary=[]
    for scene,condition,scope in sorted({(r['scene'],r['condition'],r['scope']) for r in rows}):
        group=[r for r in rows if (r['scene'],r['condition'],r['scope'])==(scene,condition,scope)]
        times={name:st.median(r['wall_ms'] for r in group if r['variant']==name) for name in functions}
        for name,ms in times.items():summary.append(dict(scene=scene,condition=condition,scope=scope,variant=name,ms=ms,
            speedup_A=times['A']/ms,speedup_B=times['B']/ms,speedup_C=times['C']/ms))
    write_json(output/'SUMMARY.json',dict(passed=True,results=summary,checks=checks,rows=len(rows),plans=plans,
        graph_misses=sum(r['graph_miss'] for r in cold),cold_calls=len(cold),cache_is_execution_plan_only=True))
    print('GRAPH_BENCHMARK_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
