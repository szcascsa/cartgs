"""冻结后才开放保留场景；逐视角配对纯渲染、真实准备和准备+渲染。"""
from harness import *
import runtime,graph_runtime
import diagnostics_check
from diagnostics_check import diagnostics


class CsvSink:
    def __init__(self,path):self.file=Path(path).open('x',newline='');self.writer=None;self.rows=0
    def add(self,row):
        if self.writer is None:self.writer=csv.DictWriter(self.file,fieldnames=list(row));self.writer.writeheader()
        self.writer.writerow(row);self.rows+=1
    def flush(self):self.file.flush()
    def close(self):self.file.close()


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def native_modules(config):
    if config.get('binaries'):
        for row in config['binaries'].values():assert digest(row['path'])==row['sha256'],row
    native=load_native('mycartgs_sparse_sixhour_v1',HERE/'build/mycartgs_sparse_sixhour_v1.so')
    graph=load_native('mycartgs_sparse_graph_study_v1',HERE/'build_graph/mycartgs_sparse_graph_study_v1.so')
    runtime.extension=lambda:native;graph_runtime.extension=lambda:graph
    diagnostics_check.extension=lambda:native


def ratio_of(condition):
    if condition.startswith('local'):return float(condition[5:])
    if condition.startswith('global'):return float(condition[6:])
    return 1. if condition=='full' else ''


def rotate(values,index):
    shift=index%len(values);result=values[shift:]+values[:shift]
    return result[::-1] if (index//len(values))%2 else result


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--round',type=int,required=True)
    parser.add_argument('--repeats',type=int,default=8);parser.add_argument('--diagnostics',action='store_true')
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--ordered',action='store_true');parser.add_argument('--device-clear',action='store_true');args=parser.parse_args()
    output=HERE/'reports'/args.output;output.mkdir(exist_ok=False)
    if args.smoke:
        config=dict(regular=dict(render_variant=5,capacity=256,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=2),
            graph=dict(plan_granularity=512,max_plans=512,ordered_coverage=args.ordered,device_clear=args.device_clear))
        config['regular']['ordered_coverage']=args.ordered
    else:
        config=json.loads((HERE/'reports/FROZEN_CANDIDATE.json').read_text())
        assert config['frozen'] and config['performance_tuning_complete']
    native_modules(config)
    sinks={name:CsvSink(output/(name+'.csv')) for name in ['samples','events','workloads','correctness','memory','startup']}
    started=time.time();checks=0;scenes=SCENES if not args.smoke else [s for s in SCENES if s['split']=='tune'][:1]
    conditions=['local'+str(r) for r in PROTOCOL['local_ratios']]+['global'+str(r) for r in PROTOCOL['global_ratios']]+[
        'fixed'+str(k) for k in [0]+PROTOCOL['fixed_requests']]+['full']
    if args.smoke:conditions=['local0.01','full']
    for scene_index,scene in enumerate(scenes):
        assert __import__('shutil').disk_usage(HERE).free>=5*(1<<30),'Disk below 5 GiB'
        tick=time.perf_counter_ns();_,source=load_scene(scene['slug']);load_ms=(time.perf_counter_ns()-tick)/1e6
        views=views_for(source,'final' if not args.smoke else 'tune')
        if args.smoke:views=views[:2]
        initial_camera=source.camera(views[0][1]);torch.cuda.synchronize()
        engines={'B':baseline_B(),'C':runtime.SparseRenderer(**config['regular']),'G':graph_runtime.GraphRenderer(**config['graph'])}
        free_before=torch.cuda.mem_get_info()[0]
        tick=time.perf_counter_ns();engines['G'].precompile(source.runtime.total,initial_camera,source.scene.background);torch.cuda.synchronize()
        compile_ms=(time.perf_counter_ns()-tick)/1e6;graph_initial=engines['G'].native.stats()
        sinks['startup'].add(dict(scene=scene['id'],round=args.round,load_ms=load_ms,graph_precompile_ms=compile_ms,
            graph_plans=graph_initial[3],graph_workspace_bytes=graph_initial[4],N=source.runtime.total,
            cuda_free_before=free_before,cuda_free_after=torch.cuda.mem_get_info()[0]))
        functions={'A':lambda cam,p:render_prepared(cam,p,source.scene.background)['render']}
        functions.update({name:(lambda cam,p,e=engine:e.render(cam,p,source.scene.background)[0]) for name,engine in engines.items()})
        for condition_index,condition in enumerate(conditions):
            load=external_load();assert not load['active'],load
            packets=[]
            for motion,frame in views:
                camera=source.camera(frame);packet,ids,stats=prepare(source,camera,frame,condition)
                ref=render_prepared(camera,packet,source.scene.background,return_radii=True)
                references={'A':ref['render']}
                for name,engine in engines.items():
                    image,radii=engine.render(camera,packet,source.scene.background)
                    equal_rgb=bits_equal(image,ref['render']);equal_radii=bits_equal(radii,ref['radii'])
                    sinks['correctness'].add(dict(scene=scene['id'],condition=condition,round=args.round,frame=frame,
                        variant=name,K=len(ids),rgb_bitwise=equal_rgb,radii_bitwise=equal_radii,
                        max_absolute_error=0. if equal_rgb else float((image-ref['render']).abs().max()),
                        max_relative_error=0. if equal_rgb else float(((image-ref['render']).abs()/ref['render'].abs().clamp_min(torch.finfo(torch.float32).tiny)).max()),
                        different_values=0 if equal_rgb else int((image.contiguous().view(torch.int32)!=ref['render'].contiguous().view(torch.int32)).sum())))
                    assert name=='B' or (equal_rgb and equal_radii),(scene['id'],condition,frame,name)
                    references[name]=image.clone()
                    checks+=1
                packets.append((motion,frame,camera,packet,ids.clone(),references))
                if args.diagnostics:
                    value=diagnostics(engines['C'],camera,packet,source.scene.background)
                    sinks['workloads'].add(dict(scene=scene['id'],split=scene['split'],condition=condition,ratio=ratio_of(condition),
                        frame=frame,motion=motion,candidate_N=stats['candidate_gaussians'],N=source.runtime.total,**value))
            base=dict(scene=scene['id'],split=scene['split'],dataset=scene['dataset_kind'],mode=scene['mode'],N=source.runtime.total,
                width=initial_camera.image_width,height=initial_camera.image_height,condition=condition,ratio=ratio_of(condition),round=args.round)
            for scope in ['raster','prepare','pipeline']:
                repeats=args.repeats if scope!='prepare' else max(2,args.repeats//2)
                for repeat in range(repeats):
                    for index,(motion,frame,camera,packet,reference_ids,reference_image) in enumerate(packets):
                        names=rotate(list(functions),index+repeat+args.round+scene_index+condition_index)
                        if scope=='prepare':names=['shared']
                        for name in names:
                            before=engines['G'].native.stats() if name=='G' else None
                            torch.cuda.synchronize();tick=time.perf_counter_ns()
                            if scope=='raster':
                                image=functions[name](camera,packet);actual_ids=reference_ids
                            else:
                                # 把每帧相机矩阵构造/上传也计入实际准备成本，不用同帧结果命中。
                                fresh_camera=source.camera(frame);fresh,actual_ids,stats=prepare(source,fresh_camera,frame,condition)
                                if scope=='pipeline':image=functions[name](fresh_camera,fresh)
                            torch.cuda.synchronize();wall_ms=(time.perf_counter_ns()-tick)/1e6
                            after=engines['G'].native.stats() if name=='G' else None
                            missed=bool(after and after[1]>before[1]);assert not missed,'Precompiled capacity missed'
                            sinks['samples'].add(dict(**base,scope=scope,variant=name,repeat=repeat,iteration=index,motion=motion,frame=frame,
                                K=len(actual_ids),wall_ms=wall_ms,plan_miss=missed))
                            if scope!='raster':
                                assert torch.equal(actual_ids,reference_ids),(scene['id'],condition,frame,'selection changed')
                                if scope=='pipeline':assert bits_equal(image,reference_image[name]),(scene['id'],condition,frame,name,'fresh prepare RGB')
                                checks+=1
            # CUDA Event单列，事件记录不进入正式同步墙钟样本。
            for view_index in [0,len(packets)-1]:
                motion,frame,camera,packet,_,_=packets[view_index]
                for name in functions:
                    values=[]
                    for iteration in range(16):
                        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                        begin.record();image=functions[name](camera,packet);end.record();end.synchronize();values.append(begin.elapsed_time(end))
                    sinks['events'].add(dict(**base,frame=frame,motion=motion,variant=name,scope='raster',
                        event_median_ms=st.median(values),event_p95_ms=float(np.quantile(values,.95))))
            for name,engine in engines.items():
                workspace=engine.native.workspace_bytes()
                torch.cuda.reset_peak_memory_stats();before=torch.cuda.memory_allocated()
                motion,frame,camera,packet,_,_=packets[-1];image=engine.render(camera,packet,source.scene.background)[0];torch.cuda.synchronize()
                sinks['memory'].add(dict(**base,variant=name,workspace_bytes=workspace,allocated_before=before,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            for sink in sinks.values():sink.flush()
            print('FINAL_PAIRED',args.round,scene['id'],condition,'checks',checks,'samples',sinks['samples'].rows,flush=True)
        assert engines['G'].native.stats()[1]==0 and engines['G'].native.stats()[2]==0
        del source,engines,functions,packets,packet,image,ref;torch.cuda.empty_cache()
    rows={name:sink.rows for name,sink in sinks.items()}
    for sink in sinks.values():sink.close()
    if config.get('binaries'):
        for row in config['binaries'].values():assert digest(row['path'])==row['sha256']
    write_json(output/'SUMMARY.json',dict(passed=True,round=args.round,scenes=len(scenes),checks=checks,rows=rows,
        elapsed_s=time.time()-started,configuration=config,fresh_pose_and_preparation_per_pipeline_request=True,
        diagnostics_outside_timing=True,no_render_images_saved=True))
    print('FINAL_PAIRED_COMPLETE',args.round,rows,flush=True)


if __name__=='__main__':main()
