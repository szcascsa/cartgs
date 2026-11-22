from common import *
from baseline import previous_runtime
from deployment_runtime import DeploymentRenderer
from large_scene import LARGE
import statistics


@torch.no_grad()
def main():
    result=[]
    for group in ['tum','replica']:
        s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
        engines={'v7':previous_runtime(s)}
        for tag,radix,tile in [('v9_off',False,0),('radix_only',True,0),('tile_only',False,4),('both',True,4)]:
            engines[tag]=DeploymentRenderer(s.model,s.gaussians,s.config,shrinking_radix=radix,compensation_tile=tile)
        for ratio in [.01,.1,.7]:
            measurements={t:{'prepare':[],'total':[]} for t in engines}
            for i in range(140):
                cam=camera(s,(i*73+19)%len(s.views));order=list(engines)
                order=order[i%len(order):]+order[:i%len(order)]
                if i%2:order.reverse()
                for tag in order:
                    torch.cuda.synchronize();start=time.perf_counter()
                    data,_,_=engines[tag].prepare(cam,ratio_budget=ratio);ready=time.perf_counter()
                    render_prepared(cam,data,s.background);torch.cuda.synchronize();end=time.perf_counter()
                    if i>=20:measurements[tag]['prepare'].append(1000*(ready-start));measurements[tag]['total'].append(1000*(end-start))
            result.append(dict(group=group,ratio=ratio,summary={tag:{stage:statistics.median(v) for stage,v in stages.items()}
                for tag,stages in measurements.items()},raw_ms=measurements))
            save(ROOT/'reports/ABLATION.json',dict(results=result));print(group,ratio,result[-1]['summary'],flush=True)
        del engines,s;torch.cuda.empty_cache()


if __name__=='__main__':main()
