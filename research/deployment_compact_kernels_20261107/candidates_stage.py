from common import *
from deployment_runtime import deployment_extension
from baseline import previous_runtime
from radix_test import timer
from large_scene import LARGE
import statistics


@torch.no_grad()
def main():
    ext=deployment_extension();checks=Comparisons('CANDIDATES_STAGE_VALIDATION.json');times=[]
    for group in ['tum','replica']:
        s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
        runtime=previous_runtime(s);b=runtime._base
        rows=torch.empty_like(b.owners);counts=torch.empty(3,device='cuda',dtype=torch.int64)
        partial=torch.empty(2*((len(b.anchors.centers)+255)//256),device='cuda',dtype=torch.int64)
        workspace=torch.empty(0,device='cuda',dtype=torch.uint8)
        for frame in [0,len(s.views)//2,len(s.views)-1]:
            cam=camera(s,frame);planes=ext._frustum_planes(cam.full_proj_transform)
            distance=b.anchors.centers@planes[:,:3].t()
            def baseline():
                nonlocal workspace
                visible=ext._frustum_test(distance,planes,b.anchors.offset_scales,runtime._reference.occupied)
                workspace=ext._visible_candidates(visible,b.owners,b.eligible_counts,rows,counts,workspace)[2]
                return visible
            def fused():
                nonlocal workspace
                visible,workspace=ext._fused_candidates(distance,planes,b.anchors.offset_scales,runtime._reference.occupied,
                    b.owners,b.eligible_counts,rows,counts,workspace,partial)
                ext._finish_counts(partial,counts)
                return visible
            visible=baseline();old_counts=counts.clone();old_rows=rows[:int(counts[0])].clone()
            actual=fused()
            checks.same(f'{group}/{frame}/visible',visible,actual)
            checks.same(f'{group}/{frame}/counts',old_counts,counts)
            checks.same(f'{group}/{frame}/rows',old_rows,rows[:int(counts[0])])
            values={'v21':[],'fused':[]}
            for repeat in range(15):
                for name,fn in [('v21',baseline),('fused',fused)][::1 if repeat%2 else -1]:values[name].append(timer(fn,30))
            row=dict(group=group,frame=frame,anchors=len(b.anchors.centers),N=int(counts[0]),
                median_us={n:statistics.median(v) for n,v in values.items()},samples=values)
            times.append(row);save(ROOT/'reports/CANDIDATES_STAGE_BENCHMARK.json',dict(results=times))
            checks.finish(group=group,frame=frame)
            print('CANDIDATE',group,frame,row['median_us'],flush=True)
        del s,runtime;torch.cuda.empty_cache()
    checks.finish(groups=2)


if __name__=='__main__':main()
