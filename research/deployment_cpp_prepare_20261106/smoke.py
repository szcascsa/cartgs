from common import *
from deployment_runtime import DeploymentRenderer


@torch.no_grad()
def main():
    s=scene();test=Comparisons('SMOKE.json')
    a=DeploymentRenderer(s.model,s.gaussians,s.config,scheduler='python')
    b=DeploymentRenderer(s.model,s.gaussians,s.config,scheduler='cpp')
    print('BUILT',flush=True)
    for frame in [0,198,396,594,791]:
        cam=camera(s,frame)
        for request in [dict(ratio_budget=r) for r in RATIOS]+[dict(budget=k) for k in [0,1,2000,5000,100000]]:
            x,ix,sx=a.prepare(cam,**request);y,iy,sy=b.prepare(cam,**request)
            test.same(str((frame,request))+'/ids',ix,iy);test.packet(str((frame,request)),x,y)
            for key in ['candidate_gaussians','eligible_candidates','requested_budget','active_gaussians','ratio','budget_shortfall']:
                assert sx[key]==sy[key],(frame,request,key,sx[key],sy[key])
        print('SMOKE',frame,len(test.checks),flush=True)
    test.finish()


if __name__=='__main__':main()
