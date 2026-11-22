"""只在诊断采集线程启动延迟、CPU/墙钟差、迁移；不把扰动计入正式收益。"""
from common import *
import gc
import importlib.util

def prototype():
    spec=importlib.util.spec_from_file_location('mycartgs_cpu_small',HERE/'development_binary/mycartgs_cpu_small.so')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    spec=importlib.util.spec_from_file_location('prototype_runtime',HERE/'source/development/small_runtime.py')
    runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime);runtime.extension=lambda:module
    return module,runtime.CpuRenderer

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);source=load_scene('Replica__office2__rgbd');records=source['records']
    cams=[camera(records[int(f*(len(records)-1))]) for f in [.19,.51,.83]];samples=[];evidence=[]
    module,cls=prototype()
    for variant,policy in [('control',0),('sleep',1),('pinned',2),('affinity',3)]:
        engine=cls(source['model'],source['pc'],source['reference'].noise,6,pool_policy=policy);engine.native.pool_trace(True)
        for condition in ['fixed0','local0.01','fixed1000']:
            kept_slow=0
            for i in range(1800):
                tick=time.perf_counter_ns();packet,ids,info=request(engine,cams[i%3],condition);elapsed=(time.perf_counter_ns()-tick)/1e6
                trace=np.asarray(engine.native.pool_traces()).reshape(-1,7)
                max_delay=float(trace[:,2].max()) if len(trace) else 0.;max_cpu_gap=float((trace[:,3]-trace[:,4]).max()) if len(trace) else 0.
                migrations=int((trace[:,5]!=trace[:,6]).sum()) if len(trace) else 0
                samples.append(dict(variant=variant,condition=condition,frame=i%3,sample=i,ms=elapsed,max_start_delay_ms=max_delay,max_cpu_gap_ms=max_cpu_gap,migrations=migrations))
                if i<3 or (elapsed>1. and kept_slow<20):
                    kept_slow+=elapsed>1.
                    for part,row in enumerate(trace):evidence.append(dict(variant=variant,condition=condition,sample=i,total_ms=elapsed,part=part,n=int(row[0]),worker=int(row[1]),start_delay_ms=row[2],wall_ms=row[3],cpu_ms=row[4],cpu_before=int(row[5]),cpu_after=int(row[6])))
            print('POOL_TRACE',variant,condition,'median',statistics.median(x['ms'] for x in samples[-1800:]),'max',max(x['ms'] for x in samples[-1800:]),flush=True)
        engine.native.pool_trace(False);del engine;gc.collect()
    write_csv(out/'samples.csv',samples);write_csv(out/'slow_evidence.csv',evidence)
    save(out/'SUMMARY.json',dict(passed=True,samples=len(samples),diagnostic_only=True,profile_overhead_included=True,
        trace_capacity_records=65536,binary_sha256=sha(module.__file__)))

if __name__=='__main__':main()
