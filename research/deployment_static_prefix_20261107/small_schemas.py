from common import *
from baseline import previous_runtime
from deployment_runtime import DeploymentRenderer


@torch.no_grad()
def main():
    test=Comparisons('SMALL_SCHEMA_VALIDATION.json');cases=0
    for group in ['both_li','selector_li','compensator_li','mode','legacy']:
        s=scene(group);old=previous_runtime(s);new=DeploymentRenderer(s.model,s.gaussians,s.config)
        for frame in [0,396,791]:
            cam=camera(s,frame)
            for k in [767,768,1000,1024,1025]:
                for local in ([False,True] if group!='legacy' else [False]):
                    for comp in [False,True]:
                        kwargs=dict(budget=k,view_conditioned=local,compensation=comp)
                        a,ia,_=old.prepare(cam,**kwargs);b,ib,_=new.prepare(cam,**kwargs)
                        label=str((group,frame,k,local,comp));test.same(label,ia,ib);test.packet(label,a,b)
                        ra=render_prepared(cam,a,s.background,return_radii=True)
                        rb=render_prepared(cam,b,s.background,return_radii=True)
                        test.same(label+'/pixels',ra['render'],rb['render']);test.same(label+'/radii',ra['radii'],rb['radii']);cases+=1
        test.finish(cases=cases,group_completed=group)
        print('SMALL_SCHEMA',group,cases,flush=True)
        del old,new,s;torch.cuda.empty_cache()
    test.finish(cases=cases,groups=5)


if __name__=='__main__':main()
