"""只在锁定的5个调优场景配对微测，记录所有候选和无效测量。"""
from harness import *
from runtime import SparseRenderer,extension


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--validation',default='screen01');parser.add_argument('--repeats',type=int,default=4)
    parser.add_argument('--candidates',default='C0,C1_shared,C2_warp,C3_keys,C13_shared_keys')
    args=parser.parse_args();output=HERE/'reports'/args.output;output.mkdir(exist_ok=False)
    quality=json.loads((HERE/'reports'/args.validation/'RESULT.json').read_text());assert quality['passed']
    configs={'C0':{},'C1_shared':dict(render_variant=1),'C2_warp':dict(render_variant=2),
        'C3_keys':dict(fused_keys=True),'C13_shared_keys':dict(render_variant=1,fused_keys=True),
        'C4_stride':dict(strided_camera=True),'C134':dict(render_variant=1,fused_keys=True,strided_camera=True),
        'C5_tasks':dict(build_variant=1),'C135_tasks':dict(render_variant=1,fused_keys=True,build_variant=1),
        'C1345':dict(render_variant=1,fused_keys=True,strided_camera=True,build_variant=1),
        'C6_sort':dict(sort_variant=1),
        'C13456':dict(render_variant=1,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=1),
        'C7_alpha2':dict(render_variant=3),'C8_alpha4':dict(render_variant=4),
        'C9_sort512':dict(sort_variant=2),
        'C13459':dict(render_variant=1,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=2),
        'Ccap64':dict(render_variant=1,capacity=64,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=2),
        'Ccap256':dict(render_variant=1,capacity=256,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=2),
        'C10_pair':dict(render_variant=5),'C11_quad':dict(render_variant=6),
        'CbestPair':dict(render_variant=5,capacity=256,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=2),
        'C12_metadata':dict(render_variant=5,capacity=256,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=2,footprint_metadata=True),
        'C13_tilefused':dict(render_variant=7,capacity=256,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=2),
        'C14_radix6':dict(render_variant=5,capacity=256,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=3),
        'C15_radix7':dict(render_variant=5,capacity=256,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=4),
        'C16_blockmerge':dict(render_variant=5,capacity=256,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=5),
        'C17_preprocess_sort':dict(render_variant=5,capacity=256,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=2,preprocess_sort=True),
        'C18_cache128':dict(render_variant=8,capacity=256,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=2),
        'C19_cache64':dict(render_variant=9,capacity=256,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=2),
        'C20_gaussianBlock':dict(render_variant=5,capacity=256,fused_keys=True,strided_camera=True,build_variant=2,sort_variant=2)}
    variants={name:SparseRenderer(**config) for name,config in configs.items() if name in quality['valid_candidates'] and name in args.candidates.split(',')}
    variants={'B':baseline_B(),**variants};rows=[];events=[];observations=[];workloads=[]
    for item in SCENES:
        if item['split']!='tune':continue
        scene,source=load_scene(item['slug'])
        frames=views_for(source)[:2]+views_for(source)[-4:]
        for condition in ['local0.01','local0.05','local0.1','local0.7','full']:
            observation=external_load();observations.append(observation);assert not observation['active'],observation
            packets=[]
            for motion,frame in frames:
                camera=source.camera(frame);packet,ids,stats=prepare(source,camera,frame,condition)
                packet=dict(sh_degree=packet['sh_degree'],inputs={k:v.detach().clone() if isinstance(v,torch.Tensor) else v for k,v in packet['inputs'].items()})
                ref=render_prepared(camera,packet,source.scene.background,return_radii=True)
                for engine in variants.values():
                    image,radii=engine.render(camera,packet,source.scene.background)
                    assert bits_equal(image,ref['render']) and bits_equal(radii,ref['radii'])
                packets.append((motion,frame,camera,packet))
                workloads.append(dict(scene=scene['id'],condition=condition,frame=frame,K=len(ids),
                    candidate_N=stats['candidate_gaussians'],visible=int((ref['radii']>0).sum()),width=camera.image_width,height=camera.image_height))
            functions={'A':lambda cam,p:render_prepared(cam,p,source.scene.background)['render']}
            functions.update({name:(lambda cam,p,e=engine:e.render(cam,p,source.scene.background)[0]) for name,engine in variants.items()})
            for _,_,cam,packet in packets:
                for function in functions.values():
                    for _ in range(3):image=function(cam,packet)
            for repeat in range(args.repeats):
                for i in range(24):
                    motion,frame,cam,packet=packets[i%len(packets)];names=list(functions)
                    shift=(i+repeat)%len(names);names=names[shift:]+names[:shift]
                    if (i+repeat)%2:names.reverse()
                    for name in names:
                        torch.cuda.synchronize();start=time.perf_counter_ns();image=functions[name](cam,packet);torch.cuda.synchronize()
                        rows.append(dict(scene=scene['id'],condition=condition,frame=frame,motion=motion,K=len(packet['inputs']['means3D']),
                            repeat=repeat,iteration=i,variant=name,wall_ms=(time.perf_counter_ns()-start)/1e6))
            _,frame,cam,packet=packets[2]
            for name,function in functions.items():
                samples=[]
                for _ in range(20):
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record();image=function(cam,packet);end.record();end.synchronize();samples.append(start.elapsed_time(end))
                events.append(dict(scene=scene['id'],condition=condition,frame=frame,variant=name,cuda_event_ms=st.median(samples)))
            observation=external_load();observations.append(observation);write_json(output/'LOAD.json',observations)
            write_csv(output/'samples.csv',rows);write_csv(output/'workloads.csv',workloads);write_csv(output/'cuda_events.csv',events)
            assert not observation['active'],observation
            print('MICRO_SCREEN',scene['id'],condition,flush=True)
        del source,packets,packet;torch.cuda.empty_cache()
    results=[]
    for scene,condition in sorted({(r['scene'],r['condition']) for r in rows}):
        group=[r for r in rows if (r['scene'],r['condition'])==(scene,condition)]
        times={name:st.median(r['wall_ms'] for r in group if r['variant']==name) for name in functions}
        for name,ms in times.items():
            results.append(dict(scene=scene,condition=condition,variant=name,ms=ms,speedup_A=times['A']/ms,speedup_B=times['B']/ms,
                repeat_speedups_B=[st.median(r['wall_ms'] for r in group if r['variant']=='B' and r['repeat']==j)/
                    st.median(r['wall_ms'] for r in group if r['variant']==name and r['repeat']==j) for j in range(args.repeats)]))
    identity=dict(candidate=str(extension().__file__),candidate_sha256=hashlib.sha256(Path(extension().__file__).read_bytes()).hexdigest())
    result=dict(passed=True,results=results,rows=len(rows),identity=identity,configs=configs,
        scope='仅调优集，GPU Event独立测量；实际切换相机，渲染输入准备在纯渲染计时外',observations=observations)
    write_json(output/'SUMMARY.json',result);print('MICRO_SCREEN_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
