"""同输入交替对比F2，外部GPU活动时整段不接受为性能证据。"""
from footprint_study import *


@torch.no_grad()
def main():
    a=argparse.ArgumentParser();a.add_argument('--output',required=True)
    a.add_argument('--modes',default='0,1,2,3')
    a.add_argument('--repeats',type=int,default=18)
    args=a.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    verify_baseline();rows=[]
    for slug in TUNE:
        s,source=load_source(slug);bg=source.scene.background
        cam=source.camera(motions(source,quick=True)[0][1])
        runs={MODES[k]:Renderer(k) for k in map(int,args.modes.split(','))}
        runs['previous_F2']=BaselineRenderer()
        for e in runs.values():e.precompile(source.runtime.total,cam,bg)
        for ci,cond in enumerate(('local0.01','local0.05','local0.1','local0.3','local0.5','full')):
            prepared=[]
            for motion,frame in motions(source,quick=True):
                cam=source.camera(frame);packet,ids,_=request(source,cam,cond)
                ref=p.render_prepared(cam,packet,bg,return_radii=True)
                for n,e in runs.items():
                    value=compare(e,cam,packet,bg,ref)
                    assert value['rgb_bitwise'] and value['radii_bitwise'],(s['id'],frame,cond,n,value)
                    for _ in range(3):e.render(cam,packet,bg)
                prepared.append((motion,frame,cam,packet,len(ids)))
            torch.cuda.synchronize()
            for rep in range(args.repeats):
                for j,(motion,frame,cam,packet,k) in enumerate(prepared):
                    for n in order(list(runs),rep+j+ci,reverse=(rep//len(runs))%2==1):
                        torch.cuda.synchronize();tick=time.perf_counter_ns()
                        image,radii=runs[n].render(cam,packet,bg)
                        torch.cuda.synchronize()
                        rows.append(dict(scene=s['id'],condition=cond,variant=n,frame=frame,repeat=rep,motion=motion,K=k,
                                         ms=(time.perf_counter_ns()-tick)*1e-6))
            print('SUBTILE_SCREEN',s['id'],cond,flush=True)
        del runs,source,prepared,packet,ref,image,radii;gc.collect();torch.cuda.empty_cache()
    append(out/'samples.csv',rows)
    groups=collections.defaultdict(list)
    for r in rows:groups[r['scene'],r['condition'],r['variant']].append(r['ms'])
    overview=[]
    for (scene,cond,name),values in groups.items():
        reference=statistics.median(groups[scene,cond,'previous_F2'])
        overview.append(dict(scene=scene,condition=cond,variant=name,p50_ms=statistics.median(values),
                             p95_ms=float(np.quantile(values,.95)),speedup=reference/statistics.median(values)))
    append(out/'summary.csv',overview)
    save(out/'DONE.json',dict(samples=len(rows),modes=args.modes,scenes=TUNE))


if __name__=='__main__':main()
