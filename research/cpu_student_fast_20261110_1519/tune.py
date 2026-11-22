"""仅三开发场景比较独立学生候选，逐节点差异与计时分开。"""
from common import *
import gc

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--variants',default='baseline,control,heads,compact,soa,vector,soa_vector');args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    conditions=['local0.1','local0.2','local0.3','local0.4','local0.5','local0.25','local0.01','local0.7','fixed0','fixed1000','fixed5000','global0.1']
    rows=[];checks=[];stages=[]
    for slug in ['Replica__office2__rgbd','Replica__room0__rgbd','Replica__office1__mono']:
        source=load_scene(slug);records=source['records'];cams=[camera(records[int(f*(len(records)-1))]) for f in [.19,.51,.83]]
        engine=native(source,'baseline',diagnose=True);truth={}
        for i,cam in enumerate(cams):
            for condition in conditions:
                packet,ids,info=request(engine,cam,condition);diagnostics=engine.native.diagnostics();truth[i,condition]=(packet,ids,diagnostics[2].clone(),diagnostics[3].clone())
        del engine;gc.collect()
        for variant in args.variants.split(','):
            engine=native(source,variant,diagnose=True)
            for condition in conditions:
                for i,cam in enumerate(cams):
                    packet,ids,info=request(engine,cam,condition);expected,oldids,scores,residual=truth[i,condition];diagnostics=engine.native.diagnostics()
                    assert torch.equal(ids,oldids) and (not len(ids) or torch.equal(scores,diagnostics[2])),(slug,variant,condition,'Selection changed')
                    for name,value in packet['inputs'].items():
                        if isinstance(value,torch.Tensor):
                            assert torch.isfinite(value).all()
                            checks.append(dict(scene=slug,variant=variant,condition=condition,frame=i,node=name,K=len(ids),same=torch.equal(value,expected['inputs'][name]),max_abs=float((value-expected['inputs'][name]).abs().max()) if value.numel() else 0.))
                    checks.append(dict(scene=slug,variant=variant,condition=condition,frame=i,node='raw_residual',K=len(ids),same=torch.equal(residual,diagnostics[3]),max_abs=float((residual-diagnostics[3]).abs().max()) if residual.numel() else 0.))
                for i in range(9):request(engine,cams[i%3],condition)
                for i in range(60):
                    tick=time.perf_counter_ns();packet,ids,info=request(engine,cams[i%3],condition);elapsed=(time.perf_counter_ns()-tick)/1e6
                    rows.append(dict(scene=slug,variant=variant,condition=condition,sample=i,N=engine.total,M=info['candidate_gaussians'],K=len(ids),ms=elapsed))
                for i in range(12):
                    _,_,info=request(engine,cams[i%3],condition,timing=True)
                    for stage,value in enumerate(info['stage_ms']):stages.append(dict(scene=slug,variant=variant,condition=condition,sample=i,stage=stage,ms=value))
                print('STUDENT_FAST_TUNE',slug,variant,condition,statistics.median(x['ms'] for x in rows[-60:]),flush=True)
            del engine,packet;gc.collect()
            for name,data in [('samples',rows),('checks',checks),('stages',stages)]:write_csv(out/(name+'.csv'),data)
        del source,truth;gc.collect()
    assert not torch.cuda.is_initialized();save(out/'SUMMARY.json',dict(passed=True,samples=len(rows),node_checks=len(checks),scenes=3,cuda_initialized=False,binary_sha256=sha(extension().__file__)))

if __name__=='__main__':main()
