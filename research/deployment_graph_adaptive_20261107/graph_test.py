from common import *
from baseline import previous_runtime
from large_scene import LARGE
from deployment_runtime import DeploymentRenderer
import statistics


@torch.no_grad()
def main():
    check=Comparisons('GRAPH_STAGE_VALIDATION.json')
    performance=[]
    for group in ['tum','replica']:
        s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
        old=previous_runtime(s)
        new=DeploymentRenderer(s.model,s.gaussians,s.config,prepare_graph=True)
        requests=[dict(budget=k) for k in [0,1,1000,2000,5000,10000,100000]]+[dict(ratio_budget=r) for r in [0.,.01,.05,.1,.3,.7,1.]]
        for frame in [0,len(s.views)//2,len(s.views)-1]:
            for request in requests:
                for local in [True,False]:
                    for compensation in [True,False]:
                        cam=camera(s,frame)
                        expected,ids,before=old.prepare(cam,**request,view_conditioned=local,compensation=compensation)
                        actual,selected,after=new.prepare(cam,**request,view_conditioned=local,compensation=compensation)
                        label=str((group,frame,request,local,compensation))
                        check.same(label+'/ids',ids,selected)
                        check.packet(label,expected,actual)
                        assert before['candidate_gaussians']==after['candidate_gaussians']
                        assert before['ratio']==after['ratio']
                        assert after['graph_count']<=28
            check.finish(group=group,frame=frame)
            print('GRAPH',group,frame,len(check.checks),flush=True)
        for request in [dict(budget=k) for k in [1000,2000,5000]]+[dict(ratio_budget=.7)]:
            times={'v13':[],'graph':[]}
            for i in range(220):
                cam=camera(s,(i*73+19)%len(s.views))
                order=[('v13',old),('graph',new)]
                if i%2:order.reverse()
                for name,engine in order:
                    torch.cuda.synchronize()
                    start=time.perf_counter()
                    engine.prepare(cam,**request)
                    torch.cuda.synchronize()
                    if i>=20:times[name].append((time.perf_counter()-start)*1000)
            performance.append(dict(group=group,request=request,median_ms={name:statistics.median(values) for name,values in times.items()},samples=times))
            save(ROOT/'reports/GRAPH_STAGE_BENCHMARK.json',dict(results=performance))
            print('PERFORMANCE',group,request,performance[-1]['median_ms'],flush=True)
        del old,new,s
        torch.cuda.empty_cache()
    check.finish(groups=2)


if __name__=='__main__':main()
