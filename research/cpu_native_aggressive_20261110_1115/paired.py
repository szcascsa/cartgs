"""跨场景两轮CPU原入口/原生准备配对；每帧重算视锥、网络及选点。"""
from cpu_study import *


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--round',type=int,required=True)
    p.add_argument('--slugs');p.add_argument('--repeats',type=int,default=4);args=p.parse_args();out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    torch.set_num_threads(6);torch.set_num_interop_threads(1)
    frozen=json.loads((HERE/'reports/FROZEN_CANDIDATE.json').read_text());assert sha(extension().__file__)==frozen['binary_sha256']
    rows=[];checks=[];stages=[];startups=[]
    specs=SCENES if not args.slugs else [s for s in SCENES if s['slug'] in args.slugs.split(',')]
    if args.round%2==0:specs=list(reversed(specs))
    for spec in specs:
        source=load_scene(spec['slug']);ref=source['reference'];tick=time.perf_counter_ns()
        engine=CpuRenderer(source['model'],source['pc'],ref.noise,6)
        startups.append(dict(scene=spec['id'],N=engine.total,init_ms=(time.perf_counter_ns()-tick)/1e6,workspace_bytes=engine.native.workspace_bytes()))
        records=source['records'];n=len(records);frame_ids=[int(.35*n),min(n-1,int(.35*n)+1),min(n-1,int(.35*n)+2),int(.07*n),int(.63*n),int(.91*n)]
        cams=[camera(records[i]) for i in frame_ids]
        conditions=['local0.01','local0.05','local0.1','local0.3','local0.7','local0.03','local0.2','fixed0','fixed1000','fixed5000','full']
        for condition_index,condition in enumerate(conditions):
            for i,cam in enumerate(cams):
                truth,ids,stats=call(ref,cam,condition);packet,selected,info=call(engine,cam,condition)
                assert len(ids)==len(selected) and stats['candidate_gaussians']==info['candidate_gaussians'],(spec['id'],condition,'budget/candidate')
                overlap=len(set(ids.tolist())&set(selected.tolist()))/max(1,len(ids))
                checks.append(dict(scene=spec['id'],round=args.round,condition=condition,frame=frame_ids[i],K=len(selected),M=stats['candidate_gaussians'],
                    same_ids=torch.equal(ids,selected),overlap=overlap,all_finite=all(torch.isfinite(v).all().item() for v in packet['inputs'].values() if isinstance(v,torch.Tensor))))
            for repeat in range(args.repeats):
                for i,cam in enumerate(cams):
                    for name in rotate(['pytorch','native'],repeat+i+condition_index+args.round):
                        e=ref if name=='pytorch' else engine
                        tick=time.perf_counter_ns();packet,ids,info=call(e,cam,condition);elapsed=(time.perf_counter_ns()-tick)/1e6
                        rows.append(dict(scene=spec['id'],slug=spec['slug'],round=args.round,condition=condition,variant=name,
                            repeat=repeat,frame=frame_ids[i],motion='continuous' if i<3 else 'jump',N=engine.total,M=info['candidate_gaussians'],K=len(ids),ms=elapsed))
            for i in [0,5]:
                if condition=='full':continue
                _,ids,info=call(engine,cams[i],condition,timing=True)
                for stage,value in enumerate(info['stage_ms']):stages.append(dict(scene=spec['id'],round=args.round,condition=condition,frame=frame_ids[i],stage=stage,ms=value))
            write_csv(out/'samples.csv',rows);write_csv(out/'checks.csv',checks);write_csv(out/'stages.csv',stages)
            print('CPU_PAIRED',args.round,spec['id'],condition,len(rows),flush=True)
        del source,ref,engine,packet,truth;write_csv(out/'startup.csv',startups)
    assert all(r['all_finite'] for r in checks)
    save(out/'SUMMARY.json',dict(passed=True,round=args.round,scenes=len(specs),samples=len(rows),checks=len(checks),
        cuda_initialized=torch.cuda.is_initialized(),binary_sha256=sha(extension().__file__),same_frame_cache=False,
        timed_scope='CPU输入相机Tensor已构造；含候选、视角、网络、严格K、补偿解码、输出分配和打包；不含渲染和H2D'))
    print('CPU_PAIRED_COMPLETE',len(rows),flush=True)

if __name__=='__main__':main()
