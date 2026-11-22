"""16组双轮，单个池驻留，未插桩总时间与重复分段计时分开。"""
from common import *
import gc

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--round',type=int,required=True);p.add_argument('--variants',default='baseline,heads,compact,vector');args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert json.loads((HERE/'reports/FROZEN_CANDIDATE.json').read_text())['binary_sha256']==sha(extension().__file__)
    rows=[];checks=[];stages=[];memory=[];specs=SCENES if args.round%2 else list(reversed(SCENES))
    for si,spec in enumerate(specs):
        source=load_scene(spec['slug']);n=len(source['records']);frames=[int(.35*n),int(.35*n)+1,int(.35*n)+2,int(.07*n),int(.63*n),int(.91*n)];cams=[camera(source['records'][f]) for f in frames];identities={};scores={};attributes={}
        for variant in rotate(args.variants.split(','),si+args.round):
            tick=time.perf_counter_ns();engine=native(source,variant)
            memory.append(dict(scene=spec['id'],variant=variant,round=args.round,N=engine.total,init_ms=(time.perf_counter_ns()-tick)/1e6,workspace_bytes=engine.native.workspace_bytes(),parameters=engine.compensator_info['parameters']))
            for condition in CONDITIONS:
                for i,cam in enumerate(cams):
                    packet,ids,info=request(engine,cam,condition);key=(condition,i);uid=hashlib.sha256(ids.numpy().tobytes()).hexdigest()
                    score=engine.native.diagnostics()[2];score_sha=hashlib.sha256(score.numpy().tobytes()).hexdigest() if condition!='full' and len(ids) else 'unused'
                    if key in identities:assert identities[key]==uid and scores[key]==score_sha
                    else:identities[key]=uid;scores[key]=score_sha
                    errors={};same=True
                    if key not in attributes:attributes[key]={name:value.clone() for name,value in packet['inputs'].items() if isinstance(value,torch.Tensor)}
                    for name,value in packet['inputs'].items():
                        if isinstance(value,torch.Tensor):
                            assert torch.isfinite(value).all();expected=attributes[key][name];error=float((value-expected).abs().max()) if value.numel() else 0.;errors['max_'+name]=error;same=same and torch.equal(value,expected)
                            if name in ['means3D','dc','shs','means2D']:assert torch.equal(value,expected),(spec['id'],variant,condition,name)
                    checks.append(dict(scene=spec['id'],variant=variant,round=args.round,condition=condition,frame=frames[i],N=engine.total,M=info['candidate_gaussians'],K=len(ids),ids_sha256=uid,score_sha256=score_sha,numerically_exact=same,finite=True,**errors))
                for block in range(3):
                    for i in range(6):request(engine,cams[i],condition)
                    for j in range(18):
                        i=j%6;tick=time.perf_counter_ns();packet,ids,info=request(engine,cams[i],condition);elapsed=(time.perf_counter_ns()-tick)/1e6
                        rows.append(dict(scene=spec['id'],slug=spec['slug'],variant=variant,round=args.round,condition=condition,block=block,sample=j,frame=frames[i],motion='continuous' if i<3 else 'jump',N=engine.total,M=info['candidate_gaussians'],K=len(ids),ms=elapsed))
                if condition!='full':
                    for j in range(24):
                        i=j%6;_,ids,info=request(engine,cams[i],condition,timing=True)
                        for stage,elapsed in enumerate(info['stage_ms']):stages.append(dict(scene=spec['id'],variant=variant,round=args.round,condition=condition,sample=j,frame=frames[i],K=len(ids),stage=stage,ms=elapsed))
            del engine,packet;gc.collect();print('STUDENT_FORMAL',args.round,spec['id'],variant,len(rows),flush=True)
        del source;gc.collect()
        for name,data in [('samples',rows),('checks',checks),('stages',stages),('memory',memory)]:write_csv(out/(name+'.csv'),data)
    assert not torch.cuda.is_initialized()
    save(out/'SUMMARY.json',dict(passed=True,round=args.round,scenes=len(specs),samples=len(rows),checks=len(checks),stage_samples=len(stages),cuda_initialized=False,score_ids_identical=True,binary_sha256=sha(extension().__file__)))

if __name__=='__main__':main()
