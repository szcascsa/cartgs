"""固定内参候选的两轮全场景配对；不重测未变化的纯Rasterizer。"""
from harness import *
from final_benchmark import native_modules,rotate,CsvSink,digest
from camera_preparation import CameraPreparer
from camera_graph_preparation import GraphCameraPreparer
from graph_runtime import GraphRenderer


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--round',type=int,required=True)
    parser.add_argument('--repeats',type=int,default=8);args=parser.parse_args()
    output=HERE/'reports'/args.output;output.mkdir(exist_ok=False)
    frozen=json.loads((HERE/'reports/FROZEN_CAMERA_GRAPH.json').read_text())
    for name,value in frozen['sources'].items():assert digest(HERE/name)==value
    config=json.loads((HERE/'reports/FROZEN_CANDIDATE.json').read_text());native_modules(config)
    sinks={name:CsvSink(output/(name+'.csv')) for name in ['samples','correctness','startup','graph_startup']}
    conditions=['local'+str(r) for r in PROTOCOL['local_ratios']]+['global'+str(r) for r in PROTOCOL['global_ratios']]+[
        'fixed'+str(k) for k in [0]+PROTOCOL['fixed_requests']]+['full']
    checks=0;started=time.time()
    for scene_index,scene in enumerate(SCENES):
        observation=external_load();assert not observation['active'],observation
        _,source=load_scene(scene['slug']);factory=CameraPreparer(MiniCam);graph_factory=GraphCameraPreparer(MiniCam);b=baseline_B();g=GraphRenderer(**config['graph'])
        views=views_for(source,'final');camera=source.camera(views[0][1]);g.precompile(source.runtime.total,camera,source.scene.background)
        torch.cuda.synchronize();tick=time.perf_counter_ns();fixed=factory.from_source(source,views[0][1]);torch.cuda.synchronize()
        sinks['startup'].add(dict(scene=scene['id'],round=args.round,first_camera_ms=(time.perf_counter_ns()-tick)/1e6,
            projection_bytes=factory.projection.storage().nbytes()))
        allocated_before=torch.cuda.memory_allocated();reserved_before=torch.cuda.memory_reserved()
        tick=time.perf_counter_ns();graph_camera=graph_factory.from_source(source,views[0][1]);torch.cuda.synchronize()
        sinks['graph_startup'].add(dict(scene=scene['id'],round=args.round,first_graph_camera_ms=(time.perf_counter_ns()-tick)/1e6,
            allocated_delta=torch.cuda.memory_allocated()-allocated_before,reserved_delta=torch.cuda.memory_reserved()-reserved_before,
            persistent_tensor_bytes=sum(t.numel()*t.element_size() for t in [graph_factory.projection,graph_factory.view,graph_factory.full,graph_factory.inverse,graph_factory.info])))
        for condition_index,condition in enumerate(conditions):
            references=[]
            for motion,frame in views:
                camera=source.camera(frame);fixed=graph_factory.from_source(source,frame)
                matrices=all(bits_equal(getattr(camera,name),getattr(fixed,name)) for name in
                    ['world_view_transform','projection_matrix','full_proj_transform','camera_center'])
                packet,ids,stats=prepare(source,camera,frame,condition)
                other,other_ids,_=prepare(source,fixed,frame,condition)
                ref=render_prepared(camera,packet,source.scene.background,return_radii=True)
                image,radii=g.render(fixed,other,source.scene.background)
                row=dict(scene=scene['id'],condition=condition,frame=frame,round=args.round,K=len(ids),
                    camera_bitwise=matrices,ids_bitwise=torch.equal(ids,other_ids),rgb_bitwise=bits_equal(image,ref['render']),
                    radii_bitwise=bits_equal(radii,ref['radii']))
                sinks['correctness'].add(row);assert all(row[name] for name in ['camera_bitwise','ids_bitwise','rgb_bitwise','radii_bitwise']),row
                references.append((motion,frame,ids.clone(),image.clone()));checks+=1
            for scope in ['pipeline','prepare']:
                for repeat in range(args.repeats if scope=='pipeline' else max(2,args.repeats//2)):
                    for index,(motion,frame,reference_ids,reference_image) in enumerate(references):
                        names=rotate(['A','B','G','F','Q'],index+repeat+args.round+scene_index+condition_index) if scope=='pipeline' else rotate(['original','fixed_intrinsics','camera_graph'],index+repeat)
                        for name in names:
                            before=g.native.stats()
                            torch.cuda.synchronize();tick=time.perf_counter_ns()
                            camera=graph_factory.from_source(source,frame) if name in ['Q','camera_graph'] else (factory.from_source(source,frame) if name in ['F','fixed_intrinsics'] else source.camera(frame))
                            packet,ids,stats=prepare(source,camera,frame,condition)
                            if scope=='pipeline':
                                if name=='A':image=render_prepared(camera,packet,source.scene.background)['render']
                                elif name=='B':image=b.render(camera,packet,source.scene.background)[0]
                                else:image=g.render(camera,packet,source.scene.background)[0]
                            torch.cuda.synchronize();elapsed=(time.perf_counter_ns()-tick)/1e6
                            after=g.native.stats();assert after[1]==before[1] and after[2]==before[2]
                            sinks['samples'].add(dict(scene=scene['id'],split=scene['split'],condition=condition,frame=frame,motion=motion,
                                round=args.round,scope=scope,variant=name,repeat=repeat,iteration=index,K=len(ids),N=source.runtime.total,wall_ms=elapsed))
                            assert torch.equal(ids,reference_ids)
                            if scope=='pipeline' and name!='B':assert bits_equal(image,reference_image)
                            checks+=1
            for sink in sinks.values():sink.flush()
            print('FINAL_CAMERA_GRAPH',args.round,scene['id'],condition,sinks['samples'].rows,flush=True)
        assert factory.rebuilds==1 and graph_factory.rebuilds==1 and g.native.stats()[1:3]==[0,0]
        del source,b,g,factory,graph_factory,references,packet,other,image,ref;torch.cuda.empty_cache()
    rows={name:sink.rows for name,sink in sinks.items()}
    for sink in sinks.values():sink.close()
    for name,value in frozen['sources'].items():assert digest(HERE/name)==value
    write_json(output/'SUMMARY.json',dict(passed=True,scenes=len(SCENES),round=args.round,checks=checks,rows=rows,
        elapsed_s=time.time()-started,projection_only_precomputed=True,pose_preparation_selection_compensation_recomputed=True,
        frozen_sources=frozen,renderer_config=config))
    print('FINAL_CAMERA_GRAPH_COMPLETE',args.round,rows,flush=True)


if __name__=='__main__':main()
