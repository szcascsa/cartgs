from common import *
from deployment_runtime import DeploymentRenderer
from baseline import previous_runtime


@torch.no_grad()
def main():
    s=scene();tag=sys.argv[1];torch.cuda.synchronize();before=torch.cuda.memory_allocated()
    runtime=previous_runtime(s) if tag=='previous_cpp' else DeploymentRenderer(s.model,s.gaussians,s.config)
    torch.cuda.synchronize();retained=torch.cuda.memory_allocated()-before;torch.cuda.reset_peak_memory_stats()
    for i in range(100):runtime.prepare(camera(s,i),ratio_budget=RATIOS[i%len(RATIOS)])
    torch.cuda.synchronize()
    save(ROOT/'reports'/f'MEMORY_{tag}.json',dict(tag=tag,retained_bytes=retained,
        peak_extra_bytes=torch.cuda.max_memory_allocated()-before,final_extra_bytes=torch.cuda.memory_allocated()-before,
        frames=100,graph=False,frustum_result_cache=False,ratio_feature_result_cache=False))


if __name__=='__main__':main()
