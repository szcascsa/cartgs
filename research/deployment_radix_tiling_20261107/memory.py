from common import *
from deployment_runtime import DeploymentRenderer
from baseline import previous_runtime
from large_scene import LARGE


@torch.no_grad()
def main():
    group=sys.argv[2] if len(sys.argv)>2 else 'tum'
    s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
    tag=sys.argv[1];torch.cuda.synchronize();before=torch.cuda.memory_allocated()
    runtime=previous_runtime(s) if tag=='v7' else DeploymentRenderer(s.model,s.gaussians,s.config)
    torch.cuda.synchronize();retained=torch.cuda.memory_allocated()-before;torch.cuda.reset_peak_memory_stats()
    for i in range(100):runtime.prepare(camera(s,i),ratio_budget=RATIOS[i%len(RATIOS)])
    torch.cuda.synchronize()
    save(ROOT/'reports'/f'MEMORY_{group}_{tag}.json',dict(tag=tag,group=group,retained_bytes=retained,
        peak_extra_bytes=torch.cuda.max_memory_allocated()-before,final_extra_bytes=torch.cuda.memory_allocated()-before,
        frames=100,graph=False,frustum_result_cache=False,ratio_feature_result_cache=False))


if __name__=='__main__':main()
