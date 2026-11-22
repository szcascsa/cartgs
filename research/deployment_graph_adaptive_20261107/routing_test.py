from common import *
from baseline import previous_runtime
from deployment_runtime import DeploymentRenderer
from large_scene import LARGE
import statistics


@torch.no_grad()
def main():
    checks=Comparisons('ROUTING_VALIDATION.json')
    results=[]
    for group in ['tum','replica']:
        s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
        engines={'v13':previous_runtime(s)}
        for graph in [False,True]:
            for dynamic in [False,True]:
                name=f'graph{int(graph)}_dynamic{int(dynamic)}'
                engines[name]=DeploymentRenderer(s.model,s.gaussians,s.config,
                    prepare_graph=graph,dynamic_compensation=dynamic)
        requests=[dict(budget=k) for k in [1,1000,2000,5000,10000]]+[
            dict(ratio_budget=r) for r in [.01,.1,.7,1.]]
        for request in requests:
            for frame in [0,len(s.views)//2,len(s.views)-1]:
                cam=camera(s,frame)
                packet,ids,_=engines['v13'].prepare(cam,**request)
                for name,engine in list(engines.items())[1:]:
                    actual,selected,stats=engine.prepare(cam,**request)
                    label=str((group,request,frame,name))
                    checks.same(label+'/ids',ids,selected)
                    checks.packet(label,packet,actual)
                    assert stats['graph_count']<=20
            times={name:[] for name in engines}
            for i in range(120):
                cam=camera(s,(i*73+19)%len(s.views))
                names=list(engines)
                names=names[i%len(names):]+names[:i%len(names)]
                if i%2:names.reverse()
                for name in names:
                    torch.cuda.synchronize()
                    start=time.perf_counter()
                    engines[name].prepare(cam,**request)
                    torch.cuda.synchronize()
                    if i>=20:times[name].append((time.perf_counter()-start)*1000)
            row=dict(group=group,request=request,median_ms={n:statistics.median(x) for n,x in times.items()},samples=times)
            results.append(row)
            save(ROOT/'reports/ROUTING_BENCHMARK.json',dict(results=results))
            print(group,request,row['median_ms'],flush=True)
        checks.finish(group=group)
        del engines,s
        torch.cuda.empty_cache()
    checks.finish(groups=2)


if __name__=='__main__':main()
