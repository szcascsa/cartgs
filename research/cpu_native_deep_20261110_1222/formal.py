"""冻结后24组双轮配对：四版本短块轮换，每帧重算，分段诊断不进正式计时。"""
from common import *
import gc

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--round',required=True,type=int);args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    frozen=json.loads((HERE/'reports/FROZEN_CANDIDATE.json').read_text());assert sha(extension().__file__)==frozen['binary_sha256']
    samples=[];checks=[];stages=[];memory=[];specs=SCENES if args.round%2 else list(reversed(SCENES))
    variants=['baseline','schedule','kernel','combined'];conditions=['local0.01','local0.05','local0.1','local0.3','local0.7','local0.03','local0.08','local0.2','fixed0','fixed1000','fixed5000','global0.1','full']
    for spec in specs:
        source=load_scene(spec['slug']);records=source['records'];n=len(records)
        frames=[int(.35*n),min(n-1,int(.35*n)+1),min(n-1,int(.35*n)+2),int(.07*n),int(.63*n),int(.91*n)];cams=[camera(records[i]) for i in frames]
        engines={}
        for name in variants:
            tick=time.perf_counter_ns();engines[name]=native(source,name)
            memory.append(dict(scene=spec['id'],variant=name,round=args.round,N=engines[name].total,init_ms=(time.perf_counter_ns()-tick)/1e6,workspace_bytes=engines[name].native.workspace_bytes()))
        for ci,condition in enumerate(conditions):
            for i,cam in enumerate(cams):
                truth,ids,stats=request(engines['baseline'],cam,condition)
                for name in variants[1:]:
                    packet,selected,info=request(engines[name],cam,condition);same=torch.equal(ids,selected)
                    assert same and info['candidate_gaussians']==stats['candidate_gaussians'],(spec['id'],condition,name,'IDs/M')
                    exact=True;maxerror=0.
                    for key,value in packet['inputs'].items():
                        if isinstance(value,torch.Tensor):
                            assert torch.isfinite(value).all();exact=exact and torch.equal(value,truth['inputs'][key])
                            if value.numel():maxerror=max(maxerror,float((value-truth['inputs'][key]).abs().max()))
                    checks.append(dict(scene=spec['id'],round=args.round,condition=condition,variant=name,frame=frames[i],K=len(ids),M=stats['candidate_gaussians'],same_ids=same,bitwise=exact,max_abs_error=maxerror))
            for block in range(3):
                for name in rotate(variants,block+ci+args.round):
                    engine=engines[name]
                    for i in range(6):request(engine,cams[i],condition)
                    for j in range(12):
                        i=j%6;tick=time.perf_counter_ns();packet,ids,info=request(engine,cams[i],condition);elapsed=(time.perf_counter_ns()-tick)/1e6
                        samples.append(dict(scene=spec['id'],slug=spec['slug'],round=args.round,variant=name,condition=condition,block=block,sample=j,frame=frames[i],
                            motion='continuous' if i<3 else 'jump',N=engine.total,M=info['candidate_gaussians'],K=len(ids),ms=elapsed))
            if condition!='full':
                for name,engine in engines.items():
                    _,_,info=request(engine,cams[0],condition,timing=True)
                    for i,value in enumerate(info['stage_ms']):stages.append(dict(scene=spec['id'],round=args.round,condition=condition,variant=name,stage=i,ms=value))
            print('DEEP_FORMAL',args.round,spec['id'],condition,len(samples),flush=True)
        write_csv(out/'samples.csv',samples);write_csv(out/'checks.csv',checks);write_csv(out/'stages.csv',stages);write_csv(out/'memory.csv',memory)
        del engines,source,truth,packet,engine;gc.collect()
    assert not torch.cuda.is_initialized()
    save(out/'SUMMARY.json',dict(passed=True,scenes=len(specs),round=args.round,samples=len(samples),checks=len(checks),all_bitwise=all(c['bitwise'] for c in checks),
        max_abs_error=max(c['max_abs_error'] for c in checks),cuda_initialized=False,binary_sha256=sha(extension().__file__)))

if __name__=='__main__':main()
