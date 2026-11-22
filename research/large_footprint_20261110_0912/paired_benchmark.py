"""冻结后全24组配对；A原入口、G上一轮、C本轮，不复用视角内容。"""
from study_common import *
from footprint_runtime import Renderer,extension
from diagnostics_check import diagnostics


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--round',type=int,required=True);parser.add_argument('--repeats',type=int,default=6)
    parser.add_argument('--diagnostics',action='store_true');args=parser.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    frozen=json.loads((HERE/'reports/FROZEN_CANDIDATE.json').read_text())
    assert frozen['frozen'] and frozen['tuning_complete']
    for name,value in frozen['sources'].items():assert digest(HERE/name)==value,name
    assert digest(frozen['binary'])==frozen['binary_sha256']
    native_modules(BASELINE_CONFIG)
    sinks={n:CsvSink(out/(n+'.csv')) for n in ['samples','events','correctness','workloads','memory','startup']}
    started=time.time();checks=0
    conditions=['local'+str(r) for r in [.01,.03,.05,.1,.3,.7]]+['global0.01','fixed0','fixed1000','fixed5000','full']
    for scene_index,scene in enumerate(SCENES):
        assert __import__('shutil').disk_usage(HERE).free>=5*(1<<30)
        observation=external_load();assert not observation['active'],observation
        _,source=load_scene(scene['slug']);views=views_for(source,'final')
        factory=AsyncGraphCameraPreparer(MiniCam);factory.from_source(source,views[0][1])
        engines={'G':previous_renderer(),'C':Renderer(frozen['mode'])}
        engines.update({name:Renderer(mode) for name,mode in frozen.get('additional_modes',{}).items()})
        camera=source.camera(views[0][1]);torch.cuda.synchronize();tick=time.perf_counter_ns()
        for engine in engines.values():engine.precompile(source.runtime.total,camera,source.scene.background)
        torch.cuda.synchronize();sinks['startup'].add(dict(scene=scene['id'],round=args.round,N=source.runtime.total,
            precompile_ms=(time.perf_counter_ns()-tick)/1e6,workspace_G=engines['G'].native.workspace_bytes(),workspace_C=engines['C'].native.workspace_bytes()))
        functions={'A':lambda cam,p:render_prepared(cam,p,source.scene.background)['render']}
        functions.update({n:(lambda cam,p,e=e:e.render(cam,p,source.scene.background)[0]) for n,e in engines.items()})
        diagnostic_engine=PreviousRegularRenderer(**BASELINE_CONFIG['regular']) if args.diagnostics else None
        for condition_index,condition in enumerate(conditions):
            packets=[]
            for motion,frame in views:
                camera=source.camera(frame);async_camera=factory.from_source(source,frame)
                assert all(bits_equal(getattr(camera,n),getattr(async_camera,n)) for n in ['world_view_transform','projection_matrix','full_proj_transform','camera_center'])
                packet,ids,stats=prepare(source,camera,frame,condition)
                other,other_ids,_=prepare(source,async_camera,frame,condition);assert torch.equal(ids,other_ids)
                ref=render_prepared(camera,packet,source.scene.background,return_radii=True)
                for name,engine in engines.items():
                    image,radii=engine.render(async_camera,other,source.scene.background)
                    row=dict(scene=scene['id'],condition=condition,round=args.round,frame=frame,variant=name,K=len(ids),
                        rgb_bitwise=bits_equal(image,ref['render']),radii_bitwise=bits_equal(radii,ref['radii']),ids_bitwise=True,camera_bitwise=True)
                    sinks['correctness'].add(row);assert row['rgb_bitwise'] and row['radii_bitwise'],row;checks+=1
                packets.append((motion,frame,camera,packet,ids.clone(),ref['render']))
                if args.diagnostics:
                    value=diagnostics(diagnostic_engine,camera,packet,source.scene.background)
                    sinks['workloads'].add(dict(scene=scene['id'],split=scene['split'],condition=condition,frame=frame,motion=motion,
                        N=source.runtime.total,candidate_N=stats['candidate_gaussians'],
                        bbox_tiles_per_visible=value['bbox_instances']/max(1,value['preprocess_visible']),**value))
            base=dict(scene=scene['id'],split=scene['split'],condition=condition,round=args.round,N=source.runtime.total)
            for scope in ['raster','pipeline','prepare']:
                for repeat in range(args.repeats if scope!='prepare' else max(2,args.repeats//2)):
                    for index,(motion,frame,camera,packet,reference_ids,reference_image) in enumerate(packets):
                        names=rotate(list(functions),index+repeat+args.round+scene_index+condition_index) if scope!='prepare' else rotate(['original','R'],index+repeat+args.round)
                        for name in names:
                            before=engines[name].native.stats() if name in engines else None
                            torch.cuda.synchronize();tick=time.perf_counter_ns()
                            if scope=='raster':image=functions[name](camera,packet);ids=reference_ids
                            else:
                                fresh_camera=source.camera(frame) if name in ['A','original'] else factory.from_source(source,frame)
                                fresh,ids,_=prepare(source,fresh_camera,frame,condition)
                                if scope=='pipeline':image=functions[name](fresh_camera,fresh)
                            torch.cuda.synchronize();wall=(time.perf_counter_ns()-tick)/1e6
                            after=engines[name].native.stats() if name in engines else None
                            assert not before or after[1:3]==before[1:3],'Unexpected graph miss/eviction'
                            sinks['samples'].add(dict(**base,scope=scope,variant=name,repeat=repeat,iteration=index,
                                frame=frame,motion=motion,K=len(ids),wall_ms=wall))
                            if scope!='raster':
                                assert torch.equal(ids,reference_ids),(scene['id'],condition,frame,'IDs')
                                if scope=='pipeline':assert bits_equal(image,reference_image),(scene['id'],condition,frame,name,'RGB')
                                checks+=1
            for view_index in [0,len(packets)-1]:
                motion,frame,camera,packet,ids,_=packets[view_index]
                for name in functions:
                    values=[]
                    for _ in range(12):
                        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                        begin.record();image=functions[name](camera,packet);end.record();end.synchronize();values.append(begin.elapsed_time(end))
                    sinks['events'].add(dict(**base,frame=frame,motion=motion,variant=name,K=len(ids),
                        event_median_ms=st.median(values),event_p95_ms=float(np.quantile(values,.95))))
            for name,engine in engines.items():
                torch.cuda.reset_peak_memory_stats();before=torch.cuda.memory_allocated()
                image=engine.render(camera,packet,source.scene.background)[0];torch.cuda.synchronize()
                sinks['memory'].add(dict(**base,variant=name,workspace_bytes=engine.native.workspace_bytes(),
                    allocated_before=before,peak_delta_bytes=torch.cuda.max_memory_allocated()-before,
                    peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            for sink in sinks.values():sink.flush()
            print('FOOTPRINT_PAIRED',args.round,scene['id'],condition,sinks['samples'].rows,flush=True)
        assert factory.rebuilds==1
        assert all(e.native.stats()[1:3]==[0,0] for e in engines.values())
        del source,engines,functions,factory,diagnostic_engine,packets,packet,other,ref,image;torch.cuda.empty_cache()
    rows={n:s.rows for n,s in sinks.items()}
    for sink in sinks.values():sink.close()
    for name,value in frozen['sources'].items():assert digest(HERE/name)==value,name
    assert digest(frozen['binary'])==frozen['binary_sha256']
    save_json(out/'SUMMARY.json',dict(passed=True,round=args.round,scenes=len(SCENES),checks=checks,rows=rows,
        elapsed_s=time.time()-started,fresh_pose_selection_compensation_each_pipeline_call=True,
        A='原相机准备+原A渲染',G='上一轮R相机准备+G渲染',C='相同R相机准备+本轮冻结渲染',no_render_images=True))
    print('FOOTPRINT_PAIRED_COMPLETE',rows,flush=True)


if __name__=='__main__':main()
