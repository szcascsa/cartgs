"""不重新选候选：增加同二进制原逻辑对照，区分编译布局与FMA复用收益。"""
from footprint_study import *
from formal import verify_frozen
from final_benchmark import native_modules

CONDITIONS=['local0.01','local0.05','local0.1','local0.3','local0.5','full']


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--round',type=int,required=True);parser.add_argument('--group',type=int,required=True)
    args=parser.parse_args();out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    f=verify_frozen();verify_baseline()
    native_modules(json.loads((previous.PRIOR/'reports/FROZEN_CANDIDATE.json').read_text()))
    scenes=SCENES[2*args.group:2*args.group+2]
    if args.round==2:scenes.reverse()
    rows=[];checks=[]
    for s in scenes:
        _,source=load_source(s['slug']);bg=source.scene.background;views=motions(source)
        if args.round==2:views.reverse()
        cam=source.camera(views[0][1])
        runs={'F2':BaselineRenderer(),'rebuilt_F2':Renderer(0),'common':Renderer(f['mode'])}
        for engine in runs.values():engine.precompile(source.runtime.total,cam,bg)
        for ci,cond in enumerate(CONDITIONS if args.round==1 else CONDITIONS[::-1]):
            cases=[]
            for motion,frame in views:
                cam=source.camera(frame);packet,ids,_=request(source,cam,cond)
                ref=p.render_prepared(cam,packet,bg,return_radii=True)
                for name,e in runs.items():
                    c=compare(e,cam,packet,bg,ref);assert c['rgb_bitwise'] and c['radii_bitwise']
                    checks.append(dict(scene=s['id'],condition=cond,frame=frame,variant=name,round=args.round,**c))
                    for _ in range(3):e.render(cam,packet,bg)
                cases.append((motion,frame,cam,packet,len(ids)))
            torch.cuda.synchronize()
            for rep in range(30):
                for j,(motion,frame,cam,packet,k) in enumerate(cases):
                    for name in order(list(runs),rep+j+ci,reverse=args.round==2):
                        torch.cuda.synchronize();tick=time.perf_counter_ns()
                        image,radii=runs[name].render(cam,packet,bg);torch.cuda.synchronize()
                        rows.append(dict(scene=s['id'],condition=cond,variant=name,round=args.round,repeat=rep,
                            frame=frame,motion=motion,N=source.runtime.total,K=k,ms=(time.perf_counter_ns()-tick)*1e-6))
            print('ATTRIBUTION_CONDITION',args.round,s['id'],cond,flush=True)
        del runs,source,cases,packet,ref,image,radii;gc.collect();torch.cuda.empty_cache()
    append(out/'samples.csv',rows);append(out/'checks.csv',checks)
    save(out/'DONE.json',dict(passed=True,samples=len(rows),checks=len(checks),no_images=True))
    verify_frozen()


if __name__=='__main__':main()
