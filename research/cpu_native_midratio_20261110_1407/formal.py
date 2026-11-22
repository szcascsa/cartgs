"""24组双轮，单个线程池驻留；独立小输出/压缩和最终组合。"""
from common import *
import gc

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--round',type=int,required=True);args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    frozen=json.loads((HERE/'reports/FROZEN_CANDIDATE.json').read_text());assert sha(extension().__file__)==frozen['binary_sha256']
    rows=[];checks=[];stages=[];memory=[];specs=SCENES if args.round%2 else list(reversed(SCENES));variants=['baseline','histogram','six','combined']
    conditions=['local0.1','local0.2','local0.3','local0.4','local0.5','local0.15','local0.25','local0.45','local0.01','local0.7','fixed0','fixed1000','fixed5000','global0.1','full']
    for si,spec in enumerate(specs):
        source=load_scene(spec['slug']);records=source['records'];n=len(records)
        frames=[int(.35*n),min(n-1,int(.35*n)+1),min(n-1,int(.35*n)+2),int(.07*n),int(.63*n),int(.91*n)];cams=[camera(records[i]) for i in frames];identities={};attributes={};rotations={};score_hashes={}
        for variant in rotate(variants,si+args.round):
            tick=time.perf_counter_ns();engine=native(source,variant)
            memory.append(dict(scene=spec['id'],variant=variant,round=args.round,N=engine.total,init_ms=(time.perf_counter_ns()-tick)/1e6,workspace_bytes=engine.native.workspace_bytes()))
            for condition in conditions:
                for i,cam in enumerate(cams):
                    packet,ids,info=request(engine,cam,condition);key=(condition,i);uid=hashlib.sha256(ids.numpy().tobytes()).hexdigest();h=hashlib.sha256()
                    for name in ['means3D','opacities','scales','rotations','dc','shs','means2D']:
                        value=packet['inputs'][name];assert torch.isfinite(value).all()
                        if name!='rotations':h.update(value.numpy().tobytes())
                    rotation=packet['inputs']['rotations'];score=engine.native.diagnostics()[2];score_sha=hashlib.sha256(score.numpy().tobytes()).hexdigest() if condition!='full' else 'full'
                    if key in identities:
                        assert uid==identities[key] and h.hexdigest()==attributes[key] and score_sha==score_hashes[key],(spec['id'],condition,variant,'Nonrotation or score mismatch')
                        rotation_error=float((rotation-rotations[key]).abs().max()) if rotation.numel() else 0.;assert rotation_error<=1e-6
                        same_rotation=torch.equal(rotation,rotations[key])
                    else:
                        identities[key]=uid;attributes[key]=h.hexdigest();rotations[key]=rotation.clone();score_hashes[key]=score_sha;rotation_error=0.;same_rotation=True
                    checks.append(dict(scene=spec['id'],round=args.round,variant=variant,condition=condition,frame=frames[i],K=len(ids),M=info['candidate_gaussians'],ids_sha256=uid,attributes_sha256=h.hexdigest(),score_sha256=score_sha,rotation_max_error=rotation_error,rotation_bitwise=same_rotation,all_finite=True))
                for block in range(3):
                    for i in range(6):request(engine,cams[i],condition)
                    for j in range(12):
                        i=j%6;tick=time.perf_counter_ns();packet,ids,info=request(engine,cams[i],condition);elapsed=(time.perf_counter_ns()-tick)/1e6
                        rows.append(dict(scene=spec['id'],slug=spec['slug'],round=args.round,variant=variant,condition=condition,block=block,sample=j,frame=frames[i],motion='continuous' if i<3 else 'jump',N=engine.total,M=info['candidate_gaussians'],K=len(ids),ms=elapsed))
                if condition!='full':
                    _,_,info=request(engine,cams[0],condition,timing=True)
                    for stage,value in enumerate(info['stage_ms']):stages.append(dict(scene=spec['id'],round=args.round,variant=variant,condition=condition,stage=stage,ms=value))
            del engine,packet;gc.collect();print('SMALL_FORMAL',args.round,spec['id'],variant,len(rows),flush=True)
        del source;gc.collect()
        for name,data in [('samples',rows),('checks',checks),('stages',stages),('memory',memory)]:write_csv(out/(name+'.csv'),data)
    assert not torch.cuda.is_initialized()
    save(out/'SUMMARY.json',dict(passed=True,round=args.round,scenes=len(specs),samples=len(rows),checks=len(checks),nonrotation_scores_bitwise=True,rotation_tolerance=1e-6,cuda_initialized=False,binary_sha256=sha(extension().__file__)))

if __name__=='__main__':main()
