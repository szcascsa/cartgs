from common import *
from deployment_runtime import deployment_extension
from baseline import previous_extension,previous_runtime
from radix_test import scratch,timer
from large_scene import LARGE
import statistics


@torch.no_grad()
def main():
    new=deployment_extension();old=previous_extension();torch.manual_seed(95)
    checks=Comparisons('SELECTION_STAGE_VALIDATION.json');results=[]
    for n in [0,1,31,255,1025,4096,8192,16384,32768,49102,65536]:
        rows=torch.randperm(2*n+17,device='cuda')[:n].contiguous()
        eligible=torch.rand(2*n+17,device='cuda')>.13
        kmax=int(eligible[rows].sum());work=scratch(n);status=torch.zeros(1,device='cuda',dtype=torch.int32)
        for kind in ['uniform','normal','ties','equal','saturated','signed_zero','extreme']:
            x=torch.rand(n,device='cuda') if kind=='uniform' else torch.randn(n,device='cuda')
            if kind=='ties':x=torch.randint(-4,4,(n,),device='cuda').float()
            if kind=='equal':x.fill_(1.)
            if kind=='saturated':x=torch.sigmoid(x*100)
            if kind=='signed_zero':x=torch.where(x<0,-torch.zeros_like(x),torch.zeros_like(x))
            if kind=='extreme':x=torch.where(x<0,-torch.finfo(torch.float32).max,torch.finfo(torch.float32).max)
            for k in sorted(set([0,min(1,kmax),int(.01*kmax),int(.7*kmax),kmax])):
                a=old._radix_select(x,rows,eligible,k,work,status)
                b=new._radix_single(x,rows,eligible,k,work,status)
                checks.same(f'{n}/{kind}/{k}',a,b)
        checks.finish(n=n)
        print('SELECT_CHECK',n,len(checks.checks),flush=True)
    for group in ['tum','replica']:
        s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
        runtime=previous_runtime(s);b=runtime._base
        for frame in [0,len(s.views)//2,len(s.views)-1]:
            cam=camera(s,frame)
            visible=b.frustum(cam.full_proj_transform,runtime._reference.occupied)
            rows,_,available=b.candidates(visible);work=scratch(len(rows));status=torch.zeros(1,device='cuda',dtype=torch.int32)
            view=old.view(b.anchors.centers,b.view_centers4,b.view_corners4,cam.camera_center.contiguous(),cam.world_view_transform,
                cam.full_proj_transform,b.model.selector_view_scene_scale,cam.image_width/cam.image_height,cam.znear)
            for ratio in [.01,.1,.7]:
                scores=old.selector(b.selector_static,rows,b.owners,view,runtime._reference.noise,runtime._selector,ratio,True,b.joint_schema)[0]
                k=min(int(ratio*len(rows)),available)
                funcs={'v21':lambda:old._radix_select(scores,rows,b.eligible,k,work,status),
                       'single':lambda:new._radix_single(scores,rows,b.eligible,k,work,status)}
                checks.same(f'{group}/{frame}/{ratio}',funcs['v21'](),funcs['single']())
                values={name:[] for name in funcs}
                for repeat in range(12):
                    for name in list(funcs)[::1 if repeat%2 else -1]:values[name].append(timer(funcs[name],20))
                row=dict(group=group,frame=frame,ratio=ratio,N=len(rows),K=k,
                    median_us={n:statistics.median(v) for n,v in values.items()},samples=values)
                results.append(row);print('SELECT_TIME',group,frame,ratio,row['median_us'],flush=True)
                save(ROOT/'reports/SELECTION_STAGE_BENCHMARK.json',dict(results=results))
        del s,runtime;torch.cuda.empty_cache()
    checks.finish(groups=2)


if __name__=='__main__':main()
