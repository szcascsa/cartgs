"""开发用固定真实输入配对，计时与分段诊断分开。"""
from cpu_study import *

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--threads',default='1,2,6');p.add_argument('--slug',default='Replica__office2__rgbd');p.add_argument('--student',action='store_true');args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);torch.set_num_threads(6);torch.set_num_interop_threads(1)
    source=load_scene(args.slug);ref=source['reference'];model=source['model'];pc=source['pc']
    frames=[int(x*(len(source['records'])-1)) for x in [.19,.51,.83]];cams=[camera(source['records'][i]) for i in frames];samples=[];stages=[]
    student=load_student(source['student_path'],'cpu')[0] if args.student else None
    for threads in map(int,args.threads.split(',')):
        for name,sel,fact in [('native',False,False),('prefix',True,False),('aggressive',True,True)]:
            engine=CpuRenderer(model,pc,ref.noise,threads,student,sel,fact)
            for condition in ['local0.01','local0.1','local0.7','fixed1000']:
                for i in range(8):packet,ids,info=call(engine,cams[i%3],condition)
                for repeat in range(36):
                    cam=cams[repeat%3];tick=time.perf_counter_ns();packet,ids,info=call(engine,cam,condition)
                    elapsed=(time.perf_counter_ns()-tick)/1e6;samples.append(dict(scene=args.slug,threads=threads,variant=name,condition=condition,repeat=repeat,M=info['candidate_gaussians'],K=len(ids),ms=elapsed,workspace_bytes=engine.native.workspace_bytes()))
                for i in range(3):
                    _,ids,info=call(engine,cams[i],condition,timing=True)
                    for j,t in enumerate(info['stage_ms']):stages.append(dict(threads=threads,variant=name,condition=condition,frame=frames[i],stage=j,ms=t))
                print('CPU_MICRO',threads,name,condition,statistics.median(x['ms'] for x in samples[-36:]),flush=True)
            del engine
            write_csv(out/'samples.csv',samples);write_csv(out/'stages.csv',stages)
    save(out/'SUMMARY.json',dict(passed=True,scene=args.slug,student=args.student,binary_sha256=sha(extension().__file__)))

if __name__=='__main__':main()
