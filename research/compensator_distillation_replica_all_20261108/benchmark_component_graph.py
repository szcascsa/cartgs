"""用现有生产Unroll内核及CUDA Graph排除Python逐次发射间隙。"""
from benchmark_paired import *


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    allowed=process_identities()
    observations=[]
    guard(allowed,observations)
    slugs=['Replica__office1__mono','Replica__office2__rgbd','Replica__room0__rgbd','Replica__room2__mono']
    rows=[]
    for slug in slugs:
        scene,source=load_scene(slug)
        student,_=load_student(ROOT/'models'/slug/'continuous_calibration/best.pt','cuda',source.fingerprint)
        run=DeploymentRenderer(source.scene.model,source.scene.gaussians,source.scene.config,student=student)
        engines=dict(teacher=source.runtime,student=run)
        b=source.runtime._base
        for frame in [19,len(source.scene.views)//2,len(source.scene.views)-20]:
            guard(allowed,observations)
            cam=source.camera(frame)
            for ratio in [.01,.05,.1,.3,.5,.7,.2]:
                for local in [False,True]:
                    case=source.sample(frame,ratio,local)
                    view=b.empty
                    if local:
                        view=run._ext.view(b.anchors.centers,b.view_centers4,b.view_corners4,
                            cam.camera_center.contiguous(),cam.world_view_transform,cam.full_proj_transform,
                            b.model.selector_view_scene_scale,cam.image_width/cam.image_height,cam.znear)
                    weights,code=ratio_controls(ratio,run.knots,b.spatial.ratio_encoding_values)
                    # 6416对应device路径的Unrolled16；大地图Host路径保留RegisterTile16。
                    tile=6416 if scene['gaussians']<=65536 else 16
                    graphs={};outputs={}
                    for name,engine in engines.items():
                        def call():
                            return engine._ext._compensate_tiled(b.offset,case['ids'],b.owners,b.reference,
                                b.tables,engine._compensator,b.compensated_master,view,engine._view_weight,
                                weights,code,local,b.joint_schema,case['softplus'],tile)
                        for _ in range(5):result=call()
                        torch.cuda.synchronize()
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):result=call()
                        graphs[name]=graph;outputs[name]=result
                        for _ in range(10):graph.replay()
                    torch.cuda.synchronize()
                    for repeat in range(3):
                        for name in order(repeat):
                            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                            torch.cuda.synchronize();start.record()
                            for _ in range(50):graphs[name].replay()
                            end.record();end.synchronize()
                            assert int(outputs[name][2])==0
                            rows.append(dict(scene=scene['id'],frame=frame,ratio=ratio,local=local,K=len(case['ids']),
                                version=name,repeat=repeat,tile=tile,cuda_ms=start.elapsed_time(end)/50))
            print('SPEED_COMPONENT_GRAPH',slug,frame,flush=True)
        del source,run,engines,b,case,graphs,outputs,student
        torch.cuda.empty_cache()
    write_csv(args.output/'samples.csv',rows)
    summary=[]
    for slug in slugs:
        scene_id=slug.replace('__','/')
        for local in [False,True]:
            for ratio in [.01,.05,.1,.3,.5,.7,.2]:
                group=[r for r in rows if r['scene']==scene_id and r['local']==local and r['ratio']==ratio]
                times={name:statistics.median(r['cuda_ms'] for r in group if r['version']==name) for name in ['teacher','student']}
                summary.append(dict(scene=scene_id,local=local,ratio=ratio,K_median=statistics.median(r['K'] for r in group),
                    **times,speedup=times['teacher']/times['student']))
    save(args.output/'summary.json',dict(passed=True,rows=len(rows),observations=observations,results=summary,
        scope='CUDA Graph replay of allocation-free captured status reset + fused gather/MLP/decode/publish; excludes selector/view generation/rasterizer',
        tile_policy='Unrolled16 on <=65536 maps, RegisterTile16 on larger maps; fixed tile control, not full adaptive dispatch',
        host_launch_gaps_reduced=True,training_or_model_changed=False))
    print('SPEED_COMPONENT_GRAPH_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
