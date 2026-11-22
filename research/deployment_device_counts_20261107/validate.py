from common import *
from baseline import previous_runtime
from deployment_runtime import DeploymentRenderer


@torch.no_grad()
def main():
    test=Comparisons('VALIDATION.json');cases=0
    for group in ['both_li','selector_li','compensator_li','mode','legacy']:
        s=scene(group);a=previous_runtime(s);b=DeploymentRenderer(s.model,s.gaussians,s.config)
        for frame in [0,198,396,594,791]:
            cam=camera(s,frame)
            options=[dict(ratio_budget=r) for r in [0.]+RATIOS+[1.]]+[dict(budget=k) for k in [0,1,7,2000,5000,100000]]
            for request in options:
                for local in ([False,True] if group!='legacy' else [False]):
                    for comp in [False,True]:
                        x,ix,sx=a.prepare(cam,**request,view_conditioned=local,compensation=comp)
                        y,iy,sy=b.prepare(cam,**request,view_conditioned=local,compensation=comp)
                        label=str((group,frame,request,local,comp))
                        test.same(label+'/ids',ix,iy);test.packet(label,x,y)
                        for key in ['candidate_gaussians','visible_anchors','eligible_candidates','requested_budget','active_gaussians','ratio','budget_shortfall']:
                            assert sx[key]==sy[key],(label,key,sx[key],sy[key])
                        rx=render_prepared(cam,x,s.background,return_radii=True);ry=render_prepared(cam,y,s.background,return_radii=True)
                        test.same(label+'/pixels',rx['render'],ry['render']);test.same(label+'/radii',rx['radii'],ry['radii']);cases+=1
            print('VALIDATED',group,frame,cases,len(test.checks),flush=True)
        # Returned outputs must survive a subsequent request, including on a non-default stream.
        cam=camera(s,396);local=group!='legacy'
        data,ids,_=b.prepare(cam,budget=1000,view_conditioned=local)
        saved={n:t.clone() for n,t in data['inputs'].items() if isinstance(t,torch.Tensor)}
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            b.prepare(camera(s,0),ratio_budget=.7,view_conditioned=local)
            other,other_ids,_=b.prepare(cam,budget=1000,view_conditioned=local)
        torch.cuda.current_stream().wait_stream(stream)
        test.same(group+'/stream/ids',ids,other_ids)
        for name,tensor in saved.items():
            test.same(group+'/ownership/'+name,tensor,data['inputs'][name]);test.same(group+'/stream/'+name,tensor,other['inputs'][name])
        test.finish(cases=cases,group_completed=group)
        del a,b,s;torch.cuda.empty_cache()
    test.finish(cases=cases,groups=5,nondefault_stream=True,output_ownership=True)


if __name__=='__main__':main()
