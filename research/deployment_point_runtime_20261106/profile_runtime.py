from common import *
from deployment_runtime import DeploymentRenderer


@torch.no_grad()
def main():
    s=scene();engines={'v3':reference_renderer(s),'deployment':DeploymentRenderer(s.model,s.gaussians,s.config)}
    for e in engines.values():
        for i in range(16):
            kwargs=dict(ratio_budget=.1,view_conditioned=True)
            if e.backend!='deployment_fp32':kwargs['measure_time']=False
            e.prepare(camera(s,396+i),**kwargs)
    for name in ['selector','compensate','view']:
        ext=engines['deployment']._ext;f=getattr(ext,name)
        def wrapped(*args,_f=f,_name=name,**kwargs):
            with torch.cuda.nvtx.range('node/deployment/'+_name):return _f(*args,**kwargs)
        setattr(ext,name,wrapped)
    for ratio in [.01,.1,.7]:
        for frame in range(396,408):
            cam=camera(s,frame);order=list(engines)
            if frame%2:order.reverse()
            for tag in order:
                kwargs=dict(ratio_budget=ratio,view_conditioned=True)
                if tag=='v3':kwargs['measure_time']=False
                with torch.cuda.nvtx.range(f'prepare/{tag}/{ratio}'):
                    data,_,_=engines[tag].prepare(cam,**kwargs)
                with torch.cuda.nvtx.range(f'render/{tag}/{ratio}'):render_prepared(cam,data,s.background)
                torch.cuda.synchronize()


if __name__=='__main__':main()
