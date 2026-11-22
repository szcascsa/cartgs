"""纯渲染双轮：F2、精确公共项候选、提前退出候选；准备不纳入计时。"""
from footprint_study import *
from final_benchmark import native_modules
from diagnostics_check import diagnostics
from runtime import SparseRenderer

CONDITIONS=[f'local{x:g}' for x in (.01,.03,.05,.1,.2,.3,.4,.5,.7)]+['fixed1000','fixed5000','full']


def verify_frozen():
    f=json.loads((HERE/'reports/FROZEN.json').read_text())
    assert f['binary_sha256']==sha(extension().__file__)
    assert all(sha(HERE/n)==h for n,h in f['sources'].items())
    return f


@torch.no_grad()
def main():
    a=argparse.ArgumentParser();a.add_argument('--output',required=True);a.add_argument('--round',type=int,required=True)
    a.add_argument('--group',type=int,required=True);a.add_argument('--repeats',type=int,default=10)
    a.add_argument('--diagnostics',action='store_true');args=a.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    f=verify_frozen();verify_baseline()
    base=json.loads((previous.PRIOR/'reports/FROZEN_CANDIDATE.json').read_text())
    native_modules(base)
    scenes=SCENES[args.group*2:args.group*2+2]
    assert scenes
    if args.round==2:scenes.reverse()
    rows,events,checks,workloads,memory=[],[],[],[],[]
    for s in scenes:
        _,source=load_source(s['slug']);bg=source.scene.background
        views=motions(source)
        if args.round==2:views.reverse()
        cam=source.camera(views[0][1])
        runs={'F2':BaselineRenderer(),'common':Renderer(f['mode']),'exit8':Renderer(f['comparison_mode'])}
        for name,run in runs.items():
            tick=time.perf_counter_ns();run.precompile(source.runtime.total,cam,bg)
            memory.append(dict(scene=s['id'],variant=name,N=source.runtime.total,
                               workspace_bytes=run.native.workspace_bytes(),plans=run.native.stats()[3],
                               startup_ms=(time.perf_counter_ns()-tick)*1e-6))
        diagnostic=SparseRenderer(**base['regular']) if args.diagnostics else None
        for ci,cond in enumerate(CONDITIONS if args.round==1 else CONDITIONS[::-1]):
            prepared=[]
            for motion,frame in views:
                cam=source.camera(frame);packet,ids,stats=request(source,cam,cond)
                ref=p.render_prepared(cam,packet,bg,return_radii=True)
                for name,e in runs.items():
                    c=compare(e,cam,packet,bg,ref)
                    checks.append(dict(scene=s['id'],condition=cond,frame=frame,variant=name,round=args.round,K=len(ids),**c))
                    assert c['rgb_bitwise'] and c['radii_bitwise'],checks[-1]
                    for _ in range(3):e.render(cam,packet,bg)
                if diagnostic is not None:
                    value=diagnostics(diagnostic,cam,packet,bg)
                    workloads.append(dict(scene=s['id'],condition=cond,frame=frame,N=source.runtime.total,
                                          M=stats['candidate_gaussians'],**value))
                prepared.append((motion,frame,cam,packet,len(ids)))
            torch.cuda.synchronize()
            for rep in range(args.repeats):
                for j,(motion,frame,cam,packet,k) in enumerate(prepared):
                    for name in order(list(runs),rep+j+ci,reverse=args.round==2):
                        before=runs[name].native.stats()
                        torch.cuda.synchronize();tick=time.perf_counter_ns()
                        image,radii=runs[name].render(cam,packet,bg);torch.cuda.synchronize()
                        ms=(time.perf_counter_ns()-tick)*1e-6
                        assert runs[name].native.stats()[1:3]==before[1:3]
                        rows.append(dict(scene=s['id'],split='tune' if s['slug'] in TUNE else 'heldout',
                                         condition=cond,variant=name,round=args.round,repeat=rep,frame=frame,motion=motion,
                                         N=source.runtime.total,K=k,ms=ms))
            for motion,frame,cam,packet,k in (prepared[0],prepared[-1]):
                for rep in range(10):
                    for name in order(list(runs),rep+ci,reverse=args.round==2):
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record();runs[name].render(cam,packet,bg);end.record();end.synchronize()
                        events.append(dict(scene=s['id'],condition=cond,variant=name,round=args.round,frame=frame,
                                           repeat=rep,K=k,cuda_ms=start.elapsed_time(end)))
            print('FORMAL_CONDITION',args.round,s['id'],cond,flush=True)
        del runs,source,prepared,packet,ref,image,radii,diagnostic;gc.collect();torch.cuda.empty_cache()
    for name,data in [('samples',rows),('checks',checks),('workloads',workloads),('events',events),('memory',memory)]:
        append(out/(name+'.csv'),data)
    save(out/'DONE.json',dict(passed=True,round=args.round,group=args.group,scenes=[s['id'] for s in scenes],
                             samples=len(rows),checks=len(checks),no_images=True))
    verify_frozen()


if __name__=='__main__':main()
