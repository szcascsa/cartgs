from common import *
from baseline import previous_runtime
from deployment_runtime import DeploymentRenderer


@torch.no_grad()
def main():
    s=scene();engines={'previous_cpp':previous_runtime(s),'cpp':DeploymentRenderer(s.model,s.gaussians,s.config)}
    for e in engines.values():
        for i in range(20):e.prepare(camera(s,396+i),ratio_budget=.1)
    for ratio in [.01,.1,.7]:
        for frame in range(396,408):
            cam=camera(s,frame);order=list(engines)
            if frame%2:order.reverse()
            for tag in order:
                with torch.cuda.nvtx.range(f'prepare/{tag}/{ratio}'):data,_,_=engines[tag].prepare(cam,ratio_budget=ratio)
                with torch.cuda.nvtx.range(f'render/{tag}/{ratio}'):render_prepared(cam,data,s.background)
                torch.cuda.synchronize()


if __name__=='__main__':main()
