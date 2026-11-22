from common import *
from baseline import previous_runtime
from deployment_runtime import DeploymentRenderer
import csv


@torch.no_grad()
def main():
    s=scene();a=previous_runtime(s);b=DeploymentRenderer(s.model,s.gaussians,s.config,scheduler='cpp')
    count=0;comparisons=0;start=time.monotonic()
    requests=[dict(ratio_budget=r) for r in RATIOS]+[dict(budget=k) for k in [0,1,1000,2000,3000,5000,10000]]
    with (ROOT/'evaluation/TRAJECTORY.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['frame','request','N','K','ids_bitwise','attributes_bitwise','pixels_bitwise','radii_bitwise']);w.writeheader()
        for frame in range(792):
            cam=camera(s,frame)
            for request in requests:
                x,ix,sx=a.prepare(cam,**request);y,iy,sy=b.prepare(cam,**request)
                assert torch.equal(ix,iy),(frame,request,'ids');comparisons+=1
                for key in x['inputs']:
                    p,q=x['inputs'][key],y['inputs'][key]
                    if p is None:assert q is None
                    else:assert torch.equal(p.contiguous().reshape(-1).view(torch.uint8),q.contiguous().reshape(-1).view(torch.uint8)),(frame,request,key);comparisons+=1
                assert sx['candidate_gaussians']==sy['candidate_gaussians'] and sx['ratio']==sy['ratio']
                rx=render_prepared(cam,x,s.background,return_radii=True);ry=render_prepared(cam,y,s.background,return_radii=True)
                assert torch.equal(rx['render'].view(torch.int32),ry['render'].view(torch.int32)),(frame,request,'pixels')
                assert torch.equal(rx['radii'],ry['radii']),(frame,request,'radii');comparisons+=2;count+=1
                w.writerow(dict(frame=frame,request=str(request),N=sy['candidate_gaussians'],K=len(iy),ids_bitwise=True,
                    attributes_bitwise=True,pixels_bitwise=True,radii_bitwise=True))
            if frame%20==0 or frame==791:
                f.flush();save(ROOT/'reports/TRAJECTORY_STATUS.json',dict(state='running',frame=frame+1,cases=count,comparisons=comparisons,seconds=time.monotonic()-start));print('TRAJECTORY',frame+1,count,flush=True)
    save(ROOT/'reports/TRAJECTORY_VALIDATION.json',dict(passed=True,frames=792,cases=count,comparisons=comparisons,images_saved=0))
    save(ROOT/'reports/TRAJECTORY_STATUS.json',dict(state='complete',frame=792,cases=count,comparisons=comparisons))


if __name__=='__main__':main()
