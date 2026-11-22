"""独占线程池复测：每次只驻留一版本，双轮反转顺序，排除多池自旋争抢。"""
from common import *
import gc

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--round',required=True,type=int);args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    frozen=json.loads((HERE/'reports/FROZEN_CANDIDATE.json').read_text());assert sha(extension().__file__)==frozen['binary_sha256']
    rows=[];checks=[];stages=[];memory=[]
    specs=SCENES if args.round%2 else list(reversed(SCENES));variants=['baseline','schedule','kernel','combined']
    conditions=['local0.01','local0.05','local0.1','local0.3','local0.7','local0.03','local0.08','local0.2','fixed0','fixed1000','fixed5000','global0.1','full']
    for si,spec in enumerate(specs):
        source=load_scene(spec['slug']);records=source['records'];n=len(records)
        frames=[int(.35*n),min(n-1,int(.35*n)+1),min(n-1,int(.35*n)+2),int(.07*n),int(.63*n),int(.91*n)];cams=[camera(records[i]) for i in frames]
        identities={};order=rotate(variants,si+args.round)
        for variant in order:
            tick=time.perf_counter_ns();engine=native(source,variant)
            memory.append(dict(scene=spec['id'],variant=variant,round=args.round,N=engine.total,init_ms=(time.perf_counter_ns()-tick)/1e6,workspace_bytes=engine.native.workspace_bytes()))
            for condition in conditions:
                for i,cam in enumerate(cams):
                    packet,ids,info=request(engine,cam,condition);digest=hashlib.sha256(ids.numpy().tobytes()).hexdigest();key=(condition,i)
                    if key in identities:assert digest==identities[key],(spec['id'],condition,variant,'IDs')
                    else:identities[key]=digest
                    checks.append(dict(scene=spec['id'],round=args.round,variant=variant,condition=condition,frame=frames[i],K=len(ids),M=info['candidate_gaussians'],ids_sha256=digest))
                for block in range(3):
                    for i in range(6):request(engine,cams[i],condition)
                    for j in range(12):
                        i=j%6;tick=time.perf_counter_ns();packet,ids,info=request(engine,cams[i],condition);elapsed=(time.perf_counter_ns()-tick)/1e6
                        rows.append(dict(scene=spec['id'],slug=spec['slug'],round=args.round,variant=variant,condition=condition,block=block,sample=j,frame=frames[i],
                            motion='continuous' if i<3 else 'jump',N=engine.total,M=info['candidate_gaussians'],K=len(ids),ms=elapsed))
                if condition!='full':
                    _,_,info=request(engine,cams[0],condition,timing=True)
                    for stage,value in enumerate(info['stage_ms']):stages.append(dict(scene=spec['id'],round=args.round,condition=condition,variant=variant,stage=stage,ms=value))
            del engine,packet;gc.collect();print('ISOLATED_FORMAL',args.round,spec['id'],variant,len(rows),flush=True)
        del source;gc.collect()
        write_csv(out/'samples.csv',rows);write_csv(out/'checks.csv',checks);write_csv(out/'stages.csv',stages);write_csv(out/'memory.csv',memory)
    assert not torch.cuda.is_initialized()
    save(out/'SUMMARY.json',dict(passed=True,round=args.round,scenes=len(specs),samples=len(rows),checks=len(checks),cuda_initialized=False,binary_sha256=sha(extension().__file__),
        protocol='一个原生线程池驻留；场景与版本顺序轮换；不删除原多池复测反例',same_ids=True))

if __name__=='__main__':main()
