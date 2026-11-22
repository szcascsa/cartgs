from common import *
from deployment_runtime import DeploymentRenderer


@torch.no_grad()
def main():
    s=scene();torch.cuda.synchronize();before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    fast=DeploymentRenderer(s.model,s.gaussians,s.config);torch.cuda.synchronize();retained=torch.cuda.memory_allocated()-before
    for i in range(100):
        fast.prepare(camera(s,i),ratio_budget=RATIOS[i%len(RATIOS)])
    torch.cuda.synchronize()
    save(ROOT/'reports/MEMORY.json',dict(retained_snapshot_bytes=retained,peak_extra_bytes=torch.cuda.max_memory_allocated()-before,
        final_extra_bytes=torch.cuda.memory_allocated()-before,request_count=100,graph_count=0,
        dynamic_feature_result_cache=False,frustum_result_cache=False))


if __name__=='__main__':main()
