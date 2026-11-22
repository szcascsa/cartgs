"""开发场景定位ATen/OpenMP与原生线程池的进程内竞争，不变更冻结核心。"""
from cpu_study import *


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    torch.set_num_threads(6);torch.set_num_interop_threads(1)
    source=load_scene('Replica__office2__rgbd');ref=source['reference']
    engine=CpuRenderer(source['model'],source['pc'],ref.noise,6)
    records=source['records'];cams=[camera(records[int(f*(len(records)-1))]) for f in [.19,.51,.83]]
    rows=[]
    for threads in [6,1,6]:
        torch.set_num_threads(threads)
        for mode in ['native_only','pytorch_only','alternate']:
            for i in range(12):
                if mode!='native_only':call(ref,cams[i%3],'local0.1')
                if mode!='pytorch_only':call(engine,cams[i%3],'local0.1')
            for i in range(60):
                cam=cams[i%3]
                for name in (['native'] if mode=='native_only' else ['pytorch'] if mode=='pytorch_only' else rotate(['native','pytorch'],i)):
                    tick=time.perf_counter_ns();call(engine if name=='native' else ref,cam,'local0.1')
                    rows.append(dict(torch_threads=threads,mode=mode,variant=name,sample=i,ms=(time.perf_counter_ns()-tick)/1e6))
            subset=[r['ms'] for r in rows if r['mode']==mode and r['torch_threads']==threads and r['variant']=='native']
            print('CONTENTION',threads,mode,statistics.median(subset) if subset else '-',flush=True)
    write_csv(out/'samples.csv',rows)
    save(out/'SUMMARY.json',dict(passed=True,parallel_info=torch.__config__.parallel_info(),
        environment={k:os.environ.get(k) for k in ['OMP_WAIT_POLICY','KMP_BLOCKTIME','GOMP_SPINCOUNT','MKL_NUM_THREADS','OMP_NUM_THREADS']},
        cuda_initialized=torch.cuda.is_initialized()))


if __name__=='__main__':main()
