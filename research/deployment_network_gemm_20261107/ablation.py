from common import *
from baseline import previous_runtime
from deployment_runtime import DeploymentRenderer
from large_scene import LARGE
import statistics


@torch.no_grad()
def main():
    results=[]
    for group in ['tum','replica']:
        s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
        engines={'v11':previous_runtime(s)}
        for name,selector,tile in [('off',False,4),('selector_only',True,4),('comp_only',False,16),('both',True,16)]:
            engines[name]=DeploymentRenderer(s.model,s.gaussians,s.config,selector_tiled=selector,compensation_tile=tile)
        for ratio in [.01,.1,.7]:
            samples={t:[] for t in engines}
            for i in range(140):
                cam=camera(s,(i*73+19)%len(s.views));order=list(engines);order=order[i%5:]+order[:i%5]
                if i%2:order.reverse()
                for tag in order:
                    torch.cuda.synchronize();start=time.perf_counter()
                    engines[tag].prepare(cam,ratio_budget=ratio);torch.cuda.synchronize();end=time.perf_counter()
                    if i>=20:samples[tag].append((end-start)*1000)
            results.append(dict(group=group,ratio=ratio,median_ms={t:statistics.median(v) for t,v in samples.items()},samples_ms=samples))
            save(ROOT/'reports/ABLATION.json',dict(results=results));print(group,ratio,results[-1]['median_ms'],flush=True)
        del s,engines;torch.cuda.empty_cache()


if __name__=='__main__':main()
