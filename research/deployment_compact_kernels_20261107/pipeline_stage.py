from common import *
from baseline import previous_runtime
from deployment_runtime import DeploymentRenderer
from large_scene import LARGE
import statistics


@torch.no_grad()
def main():
    checks=Comparisons('PIPELINE_STAGE_VALIDATION.json');results=[]
    options={'off':{},'comp6416':dict(compensation_variant=6416),
             'selection':dict(fused_selection=True),'candidates':dict(fused_candidates=True),
             'combined':dict(compensation_variant=6416,fused_selection=True,fused_candidates=True)}
    for group in ['tum','replica']:
        s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
        engines={'v21':previous_runtime(s)}
        engines.update({name:DeploymentRenderer(s.model,s.gaussians,s.config,**value) for name,value in options.items()})
        requests=[dict(budget=k) for k in [0,1,1000,2000,5000,100000]]+[
            dict(ratio_budget=r) for r in [0.,.01,.1,.7,1.]]
        for frame in [0,len(s.views)//2,len(s.views)-1]:
            cam=camera(s,frame)
            for request in requests:
                expected,ids,_=engines['v21'].prepare(cam,**request)
                for name,engine in list(engines.items())[1:]:
                    packet,selected,stats=engine.prepare(cam,**request)
                    label=f'{group}/{frame}/{request}/{name}'
                    checks.same(label+'/ids',ids,selected);checks.packet(label,expected,packet)
            checks.finish(group=group,frame=frame)
            print('PIPELINE_CHECK',group,frame,len(checks.checks),flush=True)
        for request in [dict(budget=k) for k in [1000,2000,5000]]+[dict(ratio_budget=r) for r in [.01,.1,.7]]:
            samples={name:[] for name in engines}
            for i in range(120):
                cam=camera(s,(i*73+19)%len(s.views))
                names=list(engines);names=names[i%len(names):]+names[:i%len(names)]
                if i%2:names.reverse()
                for name in names:
                    torch.cuda.synchronize();start=time.perf_counter()
                    engines[name].prepare(cam,**request);torch.cuda.synchronize()
                    if i>=20:samples[name].append((time.perf_counter()-start)*1000)
            row=dict(group=group,request=request,median_ms={n:statistics.median(v) for n,v in samples.items()},samples=samples)
            results.append(row);save(ROOT/'reports/PIPELINE_STAGE_BENCHMARK.json',dict(results=results,options=options))
            print('PIPELINE_TIME',group,request,row['median_ms'],flush=True)
        del engines,s;torch.cuda.empty_cache()
    checks.finish(groups=2)


if __name__=='__main__':main()
