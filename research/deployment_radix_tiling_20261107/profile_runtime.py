from common import *
from baseline import previous_runtime
from deployment_runtime import DeploymentRenderer
from large_scene import LARGE


@torch.no_grad()
def main():
    for group in ['tum','replica']:
        s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
        engines={'v7':previous_runtime(s),'optimized':DeploymentRenderer(s.model,s.gaussians,s.config)}
        for e in engines.values():
            for i in range(20):e.prepare(camera(s,i),ratio_budget=.1)
        for ratio in [.01,.1,.7]:
            for frame in np.linspace(0,len(s.views)-1,12,dtype=int):
                cam=camera(s,int(frame));order=list(engines)
                if frame%2:order.reverse()
                for tag in order:
                    with torch.cuda.nvtx.range(f'prepare/{group}/{tag}/{ratio}'):data,_,_=engines[tag].prepare(cam,ratio_budget=ratio)
                    with torch.cuda.nvtx.range(f'render/{group}/{tag}/{ratio}'):render_prepared(cam,data,s.background)
                    torch.cuda.synchronize()
        del engines,s;torch.cuda.empty_cache()


if __name__=='__main__':main()
