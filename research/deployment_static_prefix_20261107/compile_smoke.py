from common import *
from deployment_runtime import DeploymentRenderer,deployment_extension
from baseline import previous_runtime


@torch.no_grad()
def main():
    ext=deployment_extension()
    s=scene();old=previous_runtime(s)
    runtimes=[DeploymentRenderer(s.model,s.gaussians,s.config,selector_static_prefix=p,visible_view=v,
        small_compensation_tile=u) for p,v,u in [(False,False,False),(True,False,False),(False,True,False),(True,True,False),(True,True,True)]]
    check=Comparisons('FIRST_SMOKE.json')
    for frame in [0,396,791]:
        cam=camera(s,frame)
        for request in [dict(budget=k) for k in [0,1,17,128,1000,5000]]+[dict(ratio_budget=r) for r in [.01,.1,.7]]:
            a,ia,_=old.prepare(cam,**request)
            for j,run in enumerate(runtimes):
                b,ib,_=run.prepare(cam,**request)
                label=str((frame,request,j));check.same(label+'/ids',ia,ib);check.packet(label,a,b)
        print('SMOKE',frame,len(check.checks),flush=True)
    check.finish(binary='v30')


if __name__=='__main__':main()
