"""开发集快速独占配对；相同输入，不以组件吞吐替代当前帧墙钟。"""
from nine_study import *
from balanced_order import order


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--modes',required=True)
    parser.add_argument('--round',type=int,default=1);parser.add_argument('--nested',action='store_true')
    args=parser.parse_args();out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    rows=[];events=[]
    scenes=[s for s in SCENES if s['slug'] in TUNE]
    if args.round==2:scenes.reverse()
    for s in scenes:
        _,source=load_source(s['slug']);bg=source.scene.background
        runs={MODES[m]:Renderer(m) for m in map(int,args.modes.split(','))};runs['F2']=BaselineRenderer()
        views=motions(source,quick=True)
        if args.round==2:views.reverse()
        cam=source.camera(views[0][1])
        for e in runs.values():e.precompile(source.runtime.total,cam,bg)
        conditions=['local0.01','local0.05','local0.1','local0.2','local0.3','local0.4','local0.5','full']
        if args.round==2:conditions.reverse()
        for ci,cond in enumerate(conditions):
            cases=[]
            for motion,frame in views:
                cam=source.camera(frame)
                if args.nested:
                    full,_,_=request(source,cam,'full');packet,ids=nested_packet(full,1. if cond=='full' else float(cond[5:]))
                else:packet,ids,_=request(source,cam,cond)
                for e in runs.values():
                    for _ in range(3):e.render(cam,packet,bg)
                cases.append((motion,frame,cam,packet,len(ids)))
            torch.cuda.synchronize()
            for rep in range(max(16,len(runs)*2)):
                for j,(motion,frame,cam,packet,k) in enumerate(cases):
                    for name in order(list(runs),rep+j+ci,reverse=args.round==2):
                        torch.cuda.synchronize();tick=time.perf_counter_ns();runs[name].render(cam,packet,bg);torch.cuda.synchronize()
                        rows.append(dict(scene=s['id'],workload='nested' if args.nested else 'lod',condition=cond,variant=name,
                            frame=frame,motion=motion,round=args.round,repeat=rep,N=source.runtime.total,K=k,ms=(time.perf_counter_ns()-tick)*1e-6))
            for _,frame,cam,packet,k in (cases[0],cases[-1]):
                for rep in range(10):
                    for name in order(list(runs),rep+ci,reverse=args.round==2):
                        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        a.record();runs[name].render(cam,packet,bg);b.record();b.synchronize()
                        events.append(dict(scene=s['id'],condition=cond,variant=name,frame=frame,round=args.round,repeat=rep,K=k,cuda_ms=a.elapsed_time(b)))
            print('SCREEN_CONDITION',s['id'],cond,flush=True)
        del runs,source,cases,packet;gc.collect();torch.cuda.empty_cache()
    append(out/'samples.csv',rows);append(out/'events.csv',events)
    import pandas as pd
    t=pd.DataFrame(rows).groupby(['scene','condition','variant']).ms.agg(p50_ms='median',p95_ms=lambda x:x.quantile(.95)).reset_index()
    base=t[t.variant=='F2'][['scene','condition','p50_ms']].rename(columns={'p50_ms':'F2_ms'})
    t=t.merge(base,on=['scene','condition']);t['speedup']=t.F2_ms/t.p50_ms;t.to_csv(out/'summary.csv',index=False)
    save(out/'DONE.json',dict(passed=True,rows=len(rows),modes=args.modes,no_images=True))
    print('SCREEN_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
