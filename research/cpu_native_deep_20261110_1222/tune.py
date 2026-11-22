"""三个开发场景：每次只保留一个原生线程池，独立验证调度与微核。"""
from common import *
import gc

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--variants',default='baseline,control,schedule,kernel,combined');args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    rows=[];checks=[];stages=[];conditions=['local0.01','local0.1','local0.7','local0.08','fixed0','fixed1000','fixed5000','global0.1']
    for slug in ['Replica__office2__rgbd','Replica__room0__rgbd','TUM_RGBD__freiburg1_xyz__rgbd']:
        source=load_scene(slug);records=source['records'];cams=[camera(records[int(f*(len(records)-1))]) for f in [.19,.51,.83]]
        baseline=native(source,'baseline');truth={(i,c):request(baseline,cam,c) for i,cam in enumerate(cams) for c in conditions};del baseline;gc.collect()
        for variant in args.variants.split(','):
            engine=native(source,variant)
            for condition in conditions:
                for i,cam in enumerate(cams):
                    packet,ids,info=request(engine,cam,condition);expected,oldids,_=truth[i,condition]
                    equal=torch.equal(ids,oldids);assert equal,(slug,variant,condition,'IDs')
                    bitwise=True;maxerror=0.
                    for name,value in packet['inputs'].items():
                        if isinstance(value,torch.Tensor):
                            assert torch.isfinite(value).all(),(slug,variant,condition,name)
                            same=torch.equal(value,expected['inputs'][name]);bitwise=bitwise and same
                            if value.numel():maxerror=max(maxerror,float((value-expected['inputs'][name]).abs().max()))
                    checks.append(dict(scene=slug,variant=variant,condition=condition,frame=i,K=len(ids),same_ids=equal,bitwise=bitwise,max_abs_error=maxerror))
                for i in range(9):request(engine,cams[i%3],condition)
                for i in range(60):
                    tick=time.perf_counter_ns();packet,ids,info=request(engine,cams[i%3],condition);elapsed=(time.perf_counter_ns()-tick)/1e6
                    rows.append(dict(scene=slug,variant=variant,condition=condition,sample=i,M=info['candidate_gaussians'],K=len(ids),ms=elapsed))
                _,_,info=request(engine,cams[0],condition,timing=True)
                for stage,t in enumerate(info['stage_ms']):stages.append(dict(scene=slug,variant=variant,condition=condition,stage=stage,ms=t))
                print('DEEP_TUNE',slug,variant,condition,statistics.median(r['ms'] for r in rows[-60:]),flush=True)
            del engine;gc.collect();write_csv(out/'samples.csv',rows);write_csv(out/'checks.csv',checks);write_csv(out/'stages.csv',stages)
        del source,truth,packet,expected;gc.collect()
    save(out/'SUMMARY.json',dict(passed=True,samples=len(rows),checks=len(checks),all_bitwise=all(x['bitwise'] for x in checks),cuda_initialized=torch.cuda.is_initialized(),binary_sha256=sha(extension().__file__)))

if __name__=='__main__':main()
