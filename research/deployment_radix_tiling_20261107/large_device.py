from common import *
from baseline import previous_runtime
from deployment_runtime import DeploymentRenderer
from large_scene import LARGE
import statistics


def candidate(s):
    spec=importlib.util.spec_from_file_location('deployment_v10',ROOT/'candidate_v10/python/deployment_runtime.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.DeploymentRenderer(s.model,s.gaussians,s.config)


def retained_v9(s):
    spec=importlib.util.spec_from_file_location('deployment_v9_control',ROOT/'reference/python/deployment_runtime.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sys.path.insert(0,str(ROOT/'build/mycartgs_deployment_fp32_v9'))
    import mycartgs_deployment_fp32_v9
    module.deployment_extension=lambda:mycartgs_deployment_fp32_v9
    return module.DeploymentRenderer(s.model,s.gaussians,s.config)


@torch.no_grad()
def main():
    if '--compile-only' in sys.argv:
        spec=importlib.util.spec_from_file_location('deployment_v10',ROOT/'candidate_v10/python/deployment_runtime.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.deployment_extension()
        print('V10_COMPILED',flush=True);return
    s=BudgetScene(LARGE,None,5000,0,None,'pytorch');a=previous_runtime(s);b=candidate(s)
    check=Comparisons('LARGE_DEVICE_VALIDATION.json');cases=0
    requests=[dict(ratio_budget=r) for r in [0.,*RATIOS,1.]]+[dict(budget=k) for k in [0,1,7,1000,5000,10000,100000]]
    for frame in np.linspace(0,len(s.views)-1,9,dtype=int):
        for request in requests:
            x,ix,sx=a.prepare(camera(s,int(frame)),**request);y,iy,sy=b.prepare(camera(s,int(frame)),**request)
            label=str((int(frame),request));check.same(label+'/ids',ix,iy);check.packet(label,x,y)
            for key in ['candidate_gaussians','ratio','eligible_candidates','effective_budget']:assert sx[key]==sy[key]
            rx=render_prepared(camera(s,int(frame)),x,s.background,return_radii=True)
            ry=render_prepared(camera(s,int(frame)),y,s.background,return_radii=True)
            check.same(label+'/pixels',rx['render'],ry['render']);check.same(label+'/radii',rx['radii'],ry['radii']);cases+=1
        print('LARGE_DEVICE',frame,cases,flush=True)
    check.finish(cases=cases)
    if '--validate-only' in sys.argv:return
    engines={'v7':a,'v9_host':retained_v9(s),'v10_device':b};out=[]
    for ratio in [.01,.1,.7]:
        times={tag:{'prepare':[],'total':[]} for tag in engines}
        for i in range(170):
            cam=camera(s,(i*73+19)%len(s.views));order=list(engines);order=order[i%3:]+order[:i%3]
            if i%2:order.reverse()
            for tag in order:
                torch.cuda.synchronize();start=time.perf_counter()
                data,_,_=engines[tag].prepare(cam,ratio_budget=ratio);ready=time.perf_counter()
                render_prepared(cam,data,s.background);torch.cuda.synchronize();end=time.perf_counter()
                if i>=20:times[tag]['prepare'].append(1000*(ready-start));times[tag]['total'].append(1000*(end-start))
        row=dict(ratio=ratio,summary={tag:{stage:statistics.median(v) for stage,v in d.items()} for tag,d in times.items()},raw_ms=times)
        out.append(row);save(ROOT/'reports/LARGE_DEVICE_BENCHMARK.json',dict(results=out));print(row['ratio'],row['summary'],flush=True)


if __name__=='__main__':main()
