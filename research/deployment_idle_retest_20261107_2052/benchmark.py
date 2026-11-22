from common import *
from baseline import previous_runtime
from deployment_runtime import DeploymentRenderer
from large_scene import LARGE
import subprocess
import csv


CONDITIONS=[(str(k),dict(budget=k)) for k in [1000,2000,5000,10000]]+[
    (str(int(r*100))+'pct',dict(ratio_budget=r)) for r in [.01,.05,.1,.3,.5,.7]]+[('mixed',None)]


def gpu_state():
    fields='utilization.gpu,memory.used,clocks.sm,clocks.mem,power.draw,temperature.gpu'
    output=subprocess.check_output(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits'],text=True).strip()
    values=next(csv.reader([output]))
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
        '--format=csv,noheader,nounits'],text=True)
    processes=[dict(pid=int(row[0]),name=row[1].strip(),memory_mib=int(row[2]))
               for row in csv.reader(processes.splitlines()) if row]
    foreign=[row for row in processes if row['pid']!=os.getpid()]
    return dict(utc=time.time(),pid=os.getpid(),gpu=dict(zip(fields.split(','),[float(x) for x in values])),
                processes=processes,foreign=foreign)


def check_idle():
    state=gpu_state()
    assert not state['foreign'],('Other CUDA process appeared',state)
    return state


def request_at(fixed,i):
    options=[dict(budget=k) for k in [1,1000,2000,5000,10000]]+[
        dict(ratio_budget=r) for r in [.01,.08,.3,.7]]
    return fixed or options[i%len(options)]


def summary(values):
    return dict(median=float(np.median(values)),mean=float(np.mean(values)),
                p90=float(np.percentile(values,90)),p99=float(np.percentile(values,99)))


@torch.no_grad()
def validate():
    checks=Comparisons('VALIDATION.json');cases=0
    for group in ['tum','replica']:
        s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
        old=previous_runtime(s);new=DeploymentRenderer(s.model,s.gaussians,s.config)
        requests=[dict(budget=k) for k in [0,1,1000,2000,5000,10000,100000]]+[
            dict(ratio_budget=r) for r in [0.,.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7,1.]]
        for frame in [0,len(s.views)//2,len(s.views)-1]:
            for request in requests:
                cam=camera(s,frame);a,ia,sa=old.prepare(cam,**request);b,ib,sb=new.prepare(cam,**request)
                label=f'{group}/{frame}/{request}'
                checks.same(label+'/ids',ia,ib);checks.packet(label,a,b)
                assert sa['candidate_gaussians']==sb['candidate_gaussians'] and sa['ratio']==sb['ratio']
                x=render_prepared(cam,a,s.background,return_radii=True)
                y=render_prepared(cam,b,s.background,return_radii=True)
                checks.same(label+'/pixels',x['render'],y['render']);checks.same(label+'/radii',x['radii'],y['radii'])
                cases+=1
            check_idle()
        checks.finish(cases=cases,group=group)
        print('VALIDATED',group,cases,len(checks.checks),flush=True)
        del old,new,s;torch.cuda.empty_cache()
    checks.finish(cases=cases,groups=2)


@torch.no_grad()
def benchmark(repeat,profile=False):
    results=[]
    for group in (['tum','replica'] if repeat%2 else ['replica','tum']):
        s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
        engines={'v21':previous_runtime(s),'v27':DeploymentRenderer(s.model,s.gaussians,s.config)}
        cameras=[camera(s,(i*73+19)%len(s.views)) for i in range(180)]
        # Warm all finite execution-plan buckets outside every measured range.
        for _,fixed in CONDITIONS:
            for engine in engines.values():engine.prepare(cameras[0],**request_at(fixed,0))
        start=time.perf_counter();i=0
        while time.perf_counter()-start<2:
            for engine in engines.values():
                data,_,_=engine.prepare(cameras[i%len(cameras)],ratio_budget=.7)
                render_prepared(cameras[i%len(cameras)],data,s.background)
            i+=1
        torch.cuda.synchronize()
        conditions=CONDITIONS
        if profile:conditions=[row for row in CONDITIONS if row[0] in ['1000','5000','70pct']]
        for condition,fixed in conditions:
            before=check_idle()
            samples={n:dict(prepare_ms=[],total_ms=[],event_ms=[]) for n in engines}
            actual=[]
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            for i in range(50 if profile else 180):
                cam=cameras[i];request=request_at(fixed,i)
                names=['v21','v27'] if (i+repeat)%2 else ['v27','v21']
                for name in names:
                    torch.cuda.synchronize()
                    if profile and i>=20:torch.cuda.nvtx.range_push(f'prepare/{name}/{group}/{condition}')
                    start=time.perf_counter();begin.record()
                    packet,ids,stats=engines[name].prepare(cam,**request)
                    if profile and i>=20:torch.cuda.nvtx.range_pop()
                    end.record();end.synchronize();prepared=time.perf_counter()
                    if not profile:render_prepared(cam,packet,s.background);torch.cuda.synchronize()
                    finished=time.perf_counter()
                    if i>=20:
                        samples[name]['prepare_ms'].append((prepared-start)*1000)
                        samples[name]['total_ms'].append((finished-start)*1000)
                        samples[name]['event_ms'].append(begin.elapsed_time(end))
                        if name=='v27':actual.append(dict(frame=(i*73+19)%len(s.views),N=stats['candidate_gaussians'],K=len(ids)))
            after=check_idle()
            row=dict(group=group,condition=condition,request=fixed,actual=actual,samples=samples,
                summary={n:{k:summary(v) for k,v in values.items()} for n,values in samples.items()},gpu_before=before,gpu_after=after)
            results.append(row)
            output='PROFILE_SAMPLES.json' if profile else f'BENCHMARK_{repeat}.json'
            save(ROOT/'reports'/output,dict(repeat=repeat,profiled=profile,results=results,frames_per_condition=len(actual)))
            print(group,condition,{n:round(v['prepare_ms']['median'],5) for n,v in row['summary'].items()},flush=True)
        del engines,s;torch.cuda.empty_cache()


if __name__=='__main__':
    if sys.argv[1]=='validate':validate()
    else:benchmark(int(sys.argv[1]),'--profile' in sys.argv)
