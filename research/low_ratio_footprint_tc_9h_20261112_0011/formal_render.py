"""冻结候选的双轮纯Rasterizer；A嵌套删点与B真实LoD独立记录。"""
from nine_study import *
from balanced_order import order
from final_benchmark import native_modules
from diagnostics_check import diagnostics
from runtime import SparseRenderer

CONDITIONS=[f'local{x:g}' for x in [.01,.03,.05,.1,.2,.3,.4,.5,.7]]+['fixed0','fixed1','fixed1000','fixed5000','full']


def verify_frozen():
    f=json.loads((HERE/'reports/FROZEN_RENDER.json').read_text())
    assert sha(extension().__file__)==f['binary_sha256']
    for path,value in f['sources'].items():assert sha(HERE/path)==value,path
    return f


def candidate_workload(engine,cam,packet,bg,image,radii):
    n=len(radii);tiles=math.ceil(cam.image_width/16)*math.ceil(cam.image_height/16)
    if not n:return dict(preprocess_visible=0,alpha_participating=0,alpha_interactions=0,tile_instances=0,active_tiles=0,
        tiles=tiles,members_p50=0.,members_p95=0.,members_max=0,overflow_tiles=0,workspace_bytes=engine.native.workspace_bytes())
    check,flags,count=engine.native.diagnose()
    assert bits_equal(check,image),('diagnostic changed arithmetic',engine.mode,float((check-image).abs().max()))
    snapshot=engine.native.snapshot();counts=snapshot[3].cpu().numpy()
    return dict(preprocess_visible=int((radii>0).sum()),alpha_participating=int(flags.sum()),alpha_interactions=int(count),
        tile_instances=int(counts.sum()),active_tiles=int((counts>0).sum()),tiles=len(counts),members_p50=float(np.quantile(counts,.5)),
        members_p95=float(np.quantile(counts,.95)),members_max=int(counts.max()),overflow_tiles=int((counts>256).sum()),
        workspace_bytes=engine.native.workspace_bytes())


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--round',required=True,type=int)
    parser.add_argument('--group',required=True,type=int);parser.add_argument('--diagnostics',action='store_true');args=parser.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);f=verify_frozen()
    config=json.loads((previous.PRIOR/'reports/FROZEN_CANDIDATE.json').read_text());native_modules(config)
    rows=[];events=[];checks=[];workloads=[];memory=[]
    scenes=SCENES[args.group*2:args.group*2+2]
    if args.round==2:scenes.reverse()
    for s in scenes:
        _,source=load_source(s['slug']);bg=source.scene.background;views=motions(source)
        if args.round==2:views.reverse()
        cam=source.camera(views[0][1]);runs={MODES[m]:Renderer(m) for m in [0]+f['modes']};runs['F2']=BaselineRenderer()
        for name,e in runs.items():
            tick=time.perf_counter_ns();e.precompile(source.runtime.total,cam,bg)
            memory.append(dict(scene=s['id'],variant=name,round=args.round,N=source.runtime.total,workspace_bytes=e.native.workspace_bytes(),
                plans=e.native.stats()[3],startup_ms=(time.perf_counter_ns()-tick)*1e-6))
        diagnostic=SparseRenderer(**config['regular']) if args.diagnostics else None
        for workload in ['lod','nested']:
            conditions=CONDITIONS if args.round==1 else CONDITIONS[::-1]
            for ci,condition in enumerate(conditions):
                cases=[]
                for motion,frame in views:
                    cam=source.camera(frame)
                    if workload=='nested':
                        full,_,_=request(source,cam,'full')
                        if condition.startswith('fixed'):
                            k=min(len(full['inputs']['means3D']),int(condition[5:]));ratio=(k+.01)/max(1,len(full['inputs']['means3D']))
                        else:ratio=1. if condition=='full' else float(condition[5:])
                        packet,ids=nested_packet(full,ratio);M=source.runtime.total
                    else:packet,ids,info=request(source,cam,condition);M=info.get('candidate_gaussians',source.runtime.total)
                    ref,rr=runs['F2'].render(cam,packet,bg)
                    for name,e in runs.items():
                        image,radii=e.render(cam,packet,bg)
                        assert bits_equal(radii,rr) and bool(torch.isfinite(image).all()),(s['id'],condition,name)
                        same=bits_equal(image,ref)
                        if name=='rebuilt_F2':assert same,(s['id'],condition,name,'baseline bitwise')
                        checks.append(dict(scene=s['id'],condition=condition,workload=workload,frame=frame,variant=name,
                            round=args.round,K=len(ids),radii_bitwise=True,rgb_bitwise=same,max_abs=float((image-ref).abs().max())))
                        if args.diagnostics and name not in ['F2','rebuilt_F2']:
                            d=candidate_workload(e,cam,packet,bg,image,radii)
                            workloads.append(dict(scene=s['id'],condition=condition,workload=workload,frame=frame,variant=name,
                                N=source.runtime.total,M=M,K=len(ids),width=cam.image_width,height=cam.image_height,**d))
                        for _ in range(3):e.render(cam,packet,bg)
                    if args.diagnostics:
                        d=diagnostics(diagnostic,cam,packet,bg)
                        workloads.append(dict(scene=s['id'],condition=condition,workload=workload,frame=frame,variant='F2',N=source.runtime.total,M=M,
                            K=len(ids),width=cam.image_width,height=cam.image_height,**{k:v for k,v in d.items() if k not in ['K','width','height','bbox_instances','depth_words','nonempty_words_mean']}))
                    cases.append((motion,frame,cam,packet,len(ids),M))
                torch.cuda.synchronize()
                for repeat in range(10):
                    for j,(motion,frame,cam,packet,k,M) in enumerate(cases):
                        for name in order(list(runs),repeat+j+ci,reverse=args.round==2):
                            before=runs[name].native.stats();torch.cuda.synchronize();tick=time.perf_counter_ns()
                            runs[name].render(cam,packet,bg);torch.cuda.synchronize();ms=(time.perf_counter_ns()-tick)*1e-6
                            assert runs[name].native.stats()[1:3]==before[1:3]
                            rows.append(dict(scene=s['id'],split='tune' if s['slug'] in TUNE else 'heldout',workload=workload,
                                condition=condition,frame=frame,motion=motion,variant=name,round=args.round,repeat=repeat,N=source.runtime.total,M=M,K=k,ms=ms))
                for _,frame,cam,packet,k,M in [cases[0],cases[-1]]:
                    for repeat in range(10):
                        for name in order(list(runs),repeat+ci,reverse=args.round==2):
                            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            a.record();runs[name].render(cam,packet,bg);b.record();b.synchronize()
                            events.append(dict(scene=s['id'],condition=condition,workload=workload,frame=frame,variant=name,round=args.round,
                                repeat=repeat,K=k,cuda_ms=a.elapsed_time(b)))
                print('FORMAL_RENDER',args.round,s['id'],workload,condition,flush=True)
        del source,runs,diagnostic,cases,packet,ref,image,radii;gc.collect();torch.cuda.empty_cache()
    for name,data in [('samples',rows),('events',events),('checks',checks),('workloads',workloads),('memory',memory)]:append(out/(name+'.csv'),data)
    save(out/'DONE.json',dict(passed=True,samples=len(rows),checks=len(checks),no_images=True));verify_frozen()


if __name__=='__main__':main()
