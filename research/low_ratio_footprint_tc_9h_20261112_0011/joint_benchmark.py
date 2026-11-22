"""Teacher与匹配学生分别配对，准备/纯Raster/完整当前帧延迟分开计量。"""
from joint_study import *
from balanced_order import order
from pending_runtime import PendingRenderer

CONDITIONS=[f'local{x:g}' for x in [.01,.03,.05,.1,.2,.3,.4,.5,.7]]+['fixed0','fixed1','fixed1000','fixed5000','full']


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--group',required=True,type=int)
    parser.add_argument('--round',required=True,type=int);parser.add_argument('--pending',action='store_true')
    parser.add_argument('--native',action='store_true')
    parser.add_argument('--tune',action='store_true');parser.add_argument('--quick',action='store_true');args=parser.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);mode=json.loads((HERE/'reports/ACCEPTED_RENDER.json').read_text())['mode']
    scenes=[s for s in SCENES if not args.tune or s['slug'] in TUNE]
    scenes=scenes[args.group*2:args.group*2+2]
    if args.round==2:scenes.reverse()
    rows=[];events=[];checks=[];memory=[];identities=[]
    for spec in scenes:
        _,source=load_source(spec['slug']);student,path,payload=student_for_spec(source,spec);bg=source.scene.background
        models=[('teacher',None)]+([('student',student)] if student is not None else [])
        for kind,weights in models:
            networks=network_variants(source,weights);factory=networks['tensor_best'][1];views=motions(source,quick=args.quick)
            if args.round==2:views.reverse()
            cam=factory.from_source(source,views[0][1]);renders={'F2':BaselineRenderer(),'new':Renderer(mode)}
            variants=benchmark_variants(networks,renders)
            pending=PendingRenderer(source,weights,mode) if args.pending else None
            if pending:pending.precompile(cam,bg);variants['E_pending']=(None,None)
            native=PendingRenderer(source,weights,mode,native_dispatch=True) if args.native else None
            if native:native.precompile(cam,bg);variants['F_native']=(None,None)
            dispatch={name:engine for name,engine in [('E_pending',pending),('F_native',native)] if engine is not None}
            for r in renders.values():r.precompile(source.runtime.total,cam,bg)
            identities.append(dict(scene=spec['id'],model=kind,teacher_fingerprint=source.fingerprint,
                student=str(path) if weights is not None else None,student_sha256=sha(path) if weights is not None else None,
                selector_tensor=True,compensator_tensor=weights is not None))
            for name,(engine,_) in networks.items():
                memory.append(dict(scene=spec['id'],model=kind,variant=name,N=source.runtime.total,round=args.round,
                    projection_bytes=engine._pipeline._preparation_bytes() if name=='tensor_best' else 0,
                    tensor_bytes=engine._pipeline._tensor_bytes() if name!='fp32' else 0))
            conditions=['local0.1','local0.3','local0.5','full'] if args.quick else CONDITIONS
            for ci,condition in enumerate(conditions if args.round==1 else conditions[::-1]):
                expected={};prepared={}
                for motion,frame in views:
                    for name,(network,renderer) in variants.items():
                        if name in dispatch:continue
                        engine,camera=network
                        cam=camera.from_source(source,frame);packet,ids,stats=p.request(engine,cam,condition)
                        image,radii=renderer.render(cam,packet,bg)
                        expected[name,frame]=len(ids);prepared[name,frame]=(cam,packet,ids)
                        for _ in range(2):renderer.render(cam,packet,bg)
                    for name,engine in dispatch.items():
                        cam=factory.from_source(source,frame);image,radii,ids,stats=engine.render(cam,bg,condition)
                        refcam,refpacket,refids=prepared['D_tensor_best_new',frame]
                        refimage,refradii=renders['new'].render(refcam,refpacket,bg)
                        row=dict(scene=spec['id'],model=kind,condition=condition,frame=frame,round=args.round,variant=name,K=len(ids),
                            same_ids=torch.equal(ids,refids),rgb_bitwise=bits_equal(image,refimage),radii_bitwise=bits_equal(radii,refradii))
                        checks.append(row);assert row['same_ids'] and row['rgb_bitwise'] and row['radii_bitwise'],row
                        expected[name,frame]=len(ids)
                for repeat in range(3 if args.quick else 8):
                    for j,(motion,frame) in enumerate(views):
                        for name in order(list(variants),repeat+j+ci,reverse=args.round==2):
                            if name in dispatch:
                                torch.cuda.synchronize();tick=time.perf_counter_ns();cam=factory.from_source(source,frame)
                                image,radii,ids,stats=dispatch[name].render(cam,bg,condition);torch.cuda.synchronize()
                                rows.append(dict(scene=spec['id'],model=kind,condition=condition,variant=name,scope='pipeline',frame=frame,motion=motion,
                                    round=args.round,repeat=repeat,N=source.runtime.total,K=len(ids),ms=(time.perf_counter_ns()-tick)*1e-6))
                                assert len(ids)==expected[name,frame];continue
                            (engine,camera),renderer=variants[name]
                            for scope in ['raster','engine_prepare','prepare','pipeline']:
                                cam=camera.from_source(source,frame)
                                if scope=='raster':packet,ids,stats=p.request(engine,cam,condition)
                                torch.cuda.synchronize();tick=time.perf_counter_ns()
                                if scope in ['prepare','pipeline']:cam=camera.from_source(source,frame)
                                if scope!='raster':packet,ids,stats=p.request(engine,cam,condition)
                                if scope in ['raster','pipeline']:renderer.render(cam,packet,bg)
                                torch.cuda.synchronize();ms=(time.perf_counter_ns()-tick)*1e-6
                                assert len(ids)==expected[name,frame]
                                rows.append(dict(scene=spec['id'],model=kind,condition=condition,variant=name,scope=scope,frame=frame,motion=motion,
                                    round=args.round,repeat=repeat,N=source.runtime.total,K=len(ids),ms=ms))
                # 独立Event，不将其与CPU相机/Host阻塞墙钟混称。
                for _,frame in [views[0],views[-1]]:
                    for repeat in range(6):
                        for name in order(list(variants),repeat+ci,reverse=args.round==2):
                            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            a.record()
                            if name in dispatch:
                                cam=factory.from_source(source,frame);dispatch[name].render(cam,bg,condition)
                            else:
                                (engine,camera),renderer=variants[name];cam=camera.from_source(source,frame)
                                packet,ids,_=p.request(engine,cam,condition);renderer.render(cam,packet,bg)
                            b.record();b.synchronize();events.append(dict(scene=spec['id'],model=kind,condition=condition,variant=name,
                                frame=frame,round=args.round,repeat=repeat,cuda_ms=a.elapsed_time(b)))
                print('JOINT_FORMAL',args.round,spec['id'],kind,condition,flush=True)
                del prepared
            del networks,renders,variants,pending,native,dispatch;gc.collect();torch.cuda.empty_cache()
        del source,student;gc.collect();torch.cuda.empty_cache()
    for name,data in [('samples',rows),('events',events),('checks',checks),('memory',memory)]:append(out/(name+'.csv'),data)
    save(out/'IDENTITIES.json',identities);save(out/'DONE.json',dict(passed=True,samples=len(rows),checks=len(checks),no_images=True))


if __name__=='__main__':main()
