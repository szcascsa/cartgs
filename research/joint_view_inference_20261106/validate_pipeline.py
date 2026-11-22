"""原始独立PyTorch路径、融合和Graph的相机/选点/属性/像素逐bit对照。"""
from common import *


@torch.no_grad()
def main():
    test=Comparisons('PIPELINE_VALIDATION.json');cases=[]
    for group in ['both_li','selector_li','compensator_li','mode','legacy']:
        s=scene(group);reference=reference_renderer(s)
        variants={name:FixedBudgetRenderer(s.model,s.gaussians,s.config,0,name) for name in ['cuda','cuda_graph']}
        for frame in [0,198,396,594,791]:
            cam=camera(s,frame)
            options=[dict(ratio_budget=r) for r in RATIOS]+[dict(budget=b) for b in [0,1,17,1000,2000,5000,10000,100000]]
            options += [dict(global_ratio=r) for r in [.01,.05,.1,.3,.5,.7]]
            for request in options:
                local='global_ratio' not in request and group!='legacy'
                for compensation in [True,False]:
                    kwargs=dict(request,view_conditioned=local,compensation=compensation)
                    data,ids,stats=reference.prepare(cam,**kwargs)
                    expected=render_prepared(cam,data,s.background,return_radii=True)
                    for name,engine in variants.items():
                        for measured in [True,False]:
                            actual,chosen,out=engine.prepare(cam,measure_time=measured,**kwargs)
                            label=f'{group}/f{frame}/{request}/comp{compensation}/{name}/timed{measured}'
                            test.same(label+'/ids',chosen,ids);test.packet(label,actual,data)
                            for k in ['candidate_gaussians','eligible_candidates','requested_budget','active_gaussians','ratio','budget_shortfall']:
                                assert out[k]==stats[k],(label,k,out[k],stats[k])
                            image=render_prepared(cam,actual,s.background,return_radii=True)
                            test.same(label+'/image',image['render'],expected['render'])
                            test.same(label+'/radii',image['radii'],expected['radii'])
                            cases.append(dict(group=group,frame=frame,request=request,compensation=compensation,backend=name,measured=measured,
                                K=len(ids),N=stats['candidate_gaussians'],bitwise=True))
            print('pipeline',group,frame,'cases',len(cases),flush=True)
            test.finish(cases=cases,group_completed=group)
        del variants,reference,s;torch.cuda.empty_cache()
    test.finish(cases=cases,groups=5)


if __name__=='__main__':main()
