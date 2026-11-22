"""视角边界、稳定排序、缓存失效、Graph所有权和非默认流。"""
from common import *


@torch.no_grad()
def main():
    test=Comparisons('STATE_VALIDATION.json');s=scene();model,pc=s.model,s.gaussians
    renderer=FixedBudgetRenderer(model,pc,s.config,0,'cuda_graph');engine=renderer.inference
    generator=torch.Generator(device='cuda').manual_seed(81);events=[]
    for frame in [0,1,100,198,396,594,790,791]:
        cam=camera(s,frame)
        for near in [.0001,.01,.1,1.,10.]:
            cam.znear=near
            test.same(f'view/{frame}/near{near}',engine.view_code(cam),model.joint_view_code(cam))
        for width,height in [(1,1),(640,480),(1200,680),(680,1200)]:
            cam.image_width,cam.image_height=width,height
            test.same(f'view/{frame}/shape{width}x{height}',engine.view_code(cam),model.joint_view_code(cam))
    cam=camera(s,396);original=model.anchors.centers.clone()
    for label,value in [('coincident',cam.camera_center),('zero',torch.zeros_like(cam.camera_center))]:
        model.anchors.centers[0].copy_(value);renderer.rebuild();engine=renderer.inference
        test.same('view/'+label,engine.view_code(cam),model.joint_view_code(cam))
    model.anchors.centers.copy_(original);renderer.rebuild();engine=renderer.inference
    for n in [0,1,17,1024,13298]:
        scores=torch.randint(-4,5,(n,),device='cuda',generator=generator).float()
        if n>2:scores[0]=-0.;scores[1]=0.
        eligible=torch.rand(n,device='cuda',generator=generator)>.2;rows=torch.nonzero(eligible).flatten()
        for k in sorted(set([0,min(1,len(rows)),min(13,len(rows)),len(rows)])):
            expected=rows[scores[rows].argsort(descending=True,stable=True)[:k]].sort().values
            test.same(f'sort/{n}/{k}',engine.ext.exact_select(scores,eligible,k),expected)
    for k in [1,17,129,1000,1700,2000,5000,17]:
        for local in [False,True]:
            for j in range(4):
                cam=camera(s,[0,198,396,791][j]);code=engine.view_code(cam) if local else None
                ratio=[.01,.05,.3,.7][j];ids=torch.randperm(len(pc.get_xyz),device='cuda',generator=generator)[:k].sort().values
                engine.graph=False;expected=engine.render_data(ids,ratio,True,code,local)
                engine.graph=True;actual=engine.render_data(ids,ratio,True,code,local)
                test.packet(f'graph/{k}/{local}/{j}',actual,expected)
                saved={n:v.clone() for n,v in actual['inputs'].items() if isinstance(v,torch.Tensor)}
                engine.render_data(ids.flip(0).contiguous(),.4,True,engine.view_code(camera(s,594)) if local else None,local)
                for n,v in saved.items():test.same('output_ownership/'+n,actual['inputs'][n],v)
    assert len(engine.graphs)<=6 and engine.graph_bytes<=512*1024**2 and len(engine.shape_visits)<=32
    policy=FixedBudgetRenderer(model,pc,s.config,0,'cuda_graph')
    for _ in range(8):policy.prepare(camera(s,396),ratio_budget=.01,view_conditioned=True,measure_time=False)
    assert not policy.inference.graphs,'Dynamic ratio must not synchronously capture new shapes'
    for _ in range(4):policy.prepare(camera(s,396),1000,view_conditioned=True,measure_time=False)
    assert len(policy.inference.graphs)==1,'Stable exact point budget should capture'
    events.append(dict(graph_policy='ratio miss eager; stable requested K captures',passed=True));del policy
    torch.cuda.synchronize();other=torch.cuda.Stream();other.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(other):
        result,chosen,_=renderer.prepare(camera(s,396),2000,view_conditioned=True,measure_time=False)
    torch.cuda.current_stream().wait_stream(other)
    reference,ids,_=reference_renderer(s).prepare(camera(s,396),2000,view_conditioned=True)
    test.same('nondefault_stream/ids',chosen,ids);test.packet('nondefault_stream',result,reference)
    for label,tensor in [('master',pc._xyz),('feature',model.anchors.compensation_features_0),('weight',model.soft_net.weight),
                         ('owner',model.anchors.rows),('centers',model.anchors.centers),('scale',model.anchors.offset_scales)]:
        value=tensor.flatten()[0].clone();tensor.flatten()[0].add_(0)
        rejected=False
        try:renderer.prepare(camera(s,0),1000,view_conditioned=True)
        except RuntimeError as e:rejected='state changed' in str(e)
        assert rejected,label
        tensor.flatten()[0].copy_(value);renderer.rebuild()
        actual,selected,_=renderer.prepare(camera(s,198),1000,view_conditioned=True,measure_time=False)
        expected,ids,_=reference_renderer(s).prepare(camera(s,198),1000,view_conditioned=True)
        test.same('rebuilt/'+label+'/ids',selected,ids);test.packet('rebuilt/'+label,actual,expected)
        events.append(dict(mutation=label,stale_rejected=True,rebuild_bitwise=True))
    table=model.anchors.compensation_features_0;old=table.clone();table[0,0]=float('nan');renderer.rebuild()
    rejected=False
    try:renderer.inference.render_data(torch.nonzero(model.anchors.rows==0).flatten(),.3)
    except RuntimeError as e:rejected='Nonfinite' in str(e)
    assert rejected;table.copy_(old);renderer.rebuild()
    test.finish(events=events,finite_checks_preserved=True,graphs_bounded=True,output_ownership=True,nondefault_stream=True)
    print('state checks',len(test.checks),flush=True)


if __name__=='__main__':main()
