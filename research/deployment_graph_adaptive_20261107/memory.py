from common import *
from deployment_runtime import DeploymentRenderer
from baseline import previous_runtime
from large_scene import LARGE
import subprocess
import csv


def process_memory_mib():
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
        '--format=csv,noheader,nounits'],text=True)
    return int(next(row[1] for row in csv.reader(output.splitlines()) if int(row[0])==os.getpid()))


@torch.no_grad()
def main():
    group=sys.argv[2] if len(sys.argv)>2 else 'tum'
    s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
    tag=sys.argv[1];torch.cuda.synchronize();before=torch.cuda.memory_allocated()
    process_before=process_memory_mib()
    runtime=previous_runtime(s) if tag=='v13' else DeploymentRenderer(s.model,s.gaussians,s.config)
    torch.cuda.synchronize();retained=torch.cuda.memory_allocated()-before;torch.cuda.reset_peak_memory_stats()
    checkpoints=[]
    start=time.perf_counter()
    runtime.prepare(camera(s,0),budget=5000)
    cold_ms=(time.perf_counter()-start)*1000
    for i in range(400):
        runtime.prepare(camera(s,i%len(s.views)),ratio_budget=RATIOS[i%len(RATIOS)])
        if i in [99,199,399]:checkpoints.append(dict(requests=i+1,allocated=torch.cuda.memory_allocated()-before))
    torch.cuda.synchronize()
    process_after=process_memory_mib()
    save(ROOT/'reports'/f'MEMORY_{group}_{tag}.json',dict(tag=tag,group=group,retained_bytes=retained,
        peak_extra_bytes=torch.cuda.max_memory_allocated()-before,final_extra_bytes=torch.cuda.memory_allocated()-before,
        frames=400,cold_ms=cold_ms,checkpoints=checkpoints,graph=tag!='v13',
        process_before_mib=process_before,process_after_mib=process_after,process_delta_mib=process_after-process_before,
        frustum_result_cache=False,ratio_feature_result_cache=False))


if __name__=='__main__':main()
