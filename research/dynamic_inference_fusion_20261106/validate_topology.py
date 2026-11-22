"""真实权重的拓扑迁移与checkpoint重载；只改测试进程内存。"""
from common import *
from selector import load_selector


@torch.no_grad()
def main():
    test=Comparisons('TOPOLOGY_VALIDATION.json');s=scene();model,pc=s.model,s.gaussians
    reference=reference_renderer(s);candidate=FixedBudgetRenderer(model,pc,s.config,0,'cuda_graph');cam=camera(s,396)
    fields=['_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest'];events=[]
    def compare(label):
        for option in [dict(budget=0),dict(budget=17),dict(budget=2000),dict(ratio_budget=.01),dict(ratio_budget=.7)]:
            for comp in [False,True]:
                a,ids,stats=reference.prepare(cam,compensation=comp,view_conditioned=True,**option)
                b,chosen,out=candidate.prepare(cam,compensation=comp,view_conditioned=True,measure_time=False,**option)
                test.same(label+'/ids',chosen,ids);test.packet(label,b,a)
                assert out['ratio']==stats['ratio']
        print(label,len(pc.get_xyz),flush=True)
    def rebuild(label):
        for renderer in [reference,candidate]:
            try:renderer.prepare(cam,1000,view_conditioned=True)
            except RuntimeError as e:assert 'changed' in str(e)
            else:raise AssertionError('stale '+label)
            renderer.rebuild()
        compare(label);events.append(label)
    checkpoint=ROOT/'evaluation/selector_roundtrip.pt';torch.save(model.state_dict(),checkpoint)
    loaded=load_selector(str(checkpoint),'cuda');other=FixedBudgetRenderer(loaded,pc,s.config,0,'cuda_graph')
    a,ids,_=reference.prepare(cam,2000,view_conditioned=True)
    b,chosen,_=other.prepare(cam,2000,view_conditioned=True,measure_time=False)
    test.same('checkpoint/ids',chosen,ids);test.packet('checkpoint',b,a);del other,loaded
    compare('initial')
    pc._xyz.add_(pc._xyz.new_tensor([.01,-.02,.03]));model.anchors.centers.add_(pc._xyz.new_tensor([.01,-.02,.03]))
    rebuild('geometry_correction')
    ids=torch.arange(len(pc.get_xyz)-1,-1,-1,device='cuda')
    for n in fields:setattr(pc,n,getattr(pc,n)[ids])
    a=model.anchors;a.rows=a.rows[ids]
    if hasattr(a,'slot_inactive'):a.slot_inactive=a.slot_inactive.flatten()[ids]
    rebuild('gaussian_reorder')
    order=torch.arange(len(a.centers)-1,-1,-1,device='cuda');inverse=torch.empty_like(order);inverse[order]=torch.arange(len(order),device='cuda')
    a.rows=inverse[a.rows]
    for n in ['centers','offset_scales','keys']:setattr(a,n,getattr(a,n)[order])
    a.features=torch.nn.Parameter(a.features[order])
    for j in range(len(a.compensation_values)):
        n='compensation_features_'+str(j);setattr(a,n,torch.nn.Parameter(getattr(a,n)[order]))
    rebuild('anchor_reorder')
    parent=a.rows[:1];new_owner=len(a.centers)
    for n in ['centers','offset_scales','keys']:
        v=getattr(a,n);setattr(a,n,torch.cat((v,v[parent]),0))
    a.features=torch.nn.Parameter(torch.cat((a.features,a.features[parent]),0))
    for j in range(len(a.compensation_values)):
        n='compensation_features_'+str(j);v=getattr(a,n);setattr(a,n,torch.nn.Parameter(torch.cat((v,v[parent]),0)))
    for n in fields:
        v=getattr(pc,n);setattr(pc,n,torch.cat((v,v[:2]),0))
    pc._xyz[-1].add_(.001);a.rows=torch.cat((a.rows,a.rows[:1],a.rows.new_tensor([new_owner])))
    if hasattr(a,'slot_inactive'):a.slot_inactive=torch.cat((a.slot_inactive.flatten(),a.slot_inactive.flatten()[:2]))
    rebuild('clone_split_anchor_growing')
    keep=torch.arange(0,len(pc.get_xyz),3,device='cuda')
    for n in fields:setattr(pc,n,getattr(pc,n)[keep])
    a.rows=a.rows[keep]
    if hasattr(a,'slot_inactive'):a.slot_inactive=a.slot_inactive.flatten()[keep]
    rebuild('prune')
    for n in fields:setattr(pc,n,getattr(pc,n).repeat((24,)+(1,)*(getattr(pc,n).ndim-1)))
    a.rows=a.rows.repeat(24)
    if hasattr(a,'slot_inactive'):a.slot_inactive=a.slot_inactive.flatten().repeat(24)
    rebuild('large_N')
    for n in fields:setattr(pc,n,getattr(pc,n)[:0])
    a.rows=a.rows[:0]
    if hasattr(a,'slot_inactive'):a.slot_inactive=a.slot_inactive.flatten()[:0]
    rebuild('empty_topology')
    test.finish(events=events,checkpoint_reload=True,synthetic_large_N=106416)


if __name__=='__main__':main()
