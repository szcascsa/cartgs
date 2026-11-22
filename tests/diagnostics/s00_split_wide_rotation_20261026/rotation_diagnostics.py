"""只读检查基础Rotation分布、新输入权重更新及选中K行的输出依赖。"""
import json
import torch
from run import ROOT, PROTOCOL, Network, config, canonical_rotation, rt


@torch.no_grad()
def main():
    scene=rt.Scene('fr1_xyz')
    q=canonical_rotation(scene.g.get_rotation.detach())
    rt.save_json(ROOT/'rotation_input_statistics.json',dict(
        N=scene.N,normalized=True,sign_rule='largest absolute component positive',
        mean=q.mean(0).cpu().tolist(),std=q.std(0).cpu().tolist(),
        p10=torch.quantile(q,.1,dim=0).cpu().tolist(),p90=torch.quantile(q,.9,dim=0).cpu().tolist(),
        sign_invariant=torch.equal(q,canonical_rotation(-scene.g.get_rotation.detach())),
        max_norm_error=float((q.norm(dim=1)-1).abs().max())))
    rows=[]
    for seed in PROTOCOL['seeds']:
        cfg=config('split4_rotation')
        path=ROOT/'results/fr1_xyz'/cfg['id']/str(seed)/'step_12000.pt'
        payload=torch.load(path,map_location='cuda')
        net=Network(scene,cfg,seed)
        net.load_state_dict(payload['state_dict'])
        stats={}
        for name in ['xyz','scale','rotation']:
            weight=net.compensation.branches[name][0].weight
            stats[name]=dict(rotation_columns_norm=float(weight[:,59:].norm()),
                previous_input_columns_norm=float(weight[:,:59].norm()))
        rt.save_json(ROOT/f'rotation_input_weight_updates_{seed}.json',stats)
        for ratio in rt.RATIOS:
            p,h,logits=net(ratio)
            margin=logits.double()-torch.logit(scene.uniform.double())
            margin[~scene.eligible]=-torch.inf
            ids=torch.argsort(margin,descending=True,stable=True)[:scene.K[ratio]].sort().values
            actual=net.compensate(h,ids,ratio)
            old=net.base_rotation
            net.base_rotation=torch.zeros_like(old)
            neutral=net.compensate(h,ids,ratio)
            net.base_rotation=old
            change=actual-neutral
            assert torch.count_nonzero(change[:,10:])==0
            for name,start,end in [('xyz',0,3),('scale',3,6),('rotation',6,10)]:
                rows.append(dict(seed=seed,ratio=ratio,group=name,selected=len(ids),
                    output_change_rms=float(change[:,start:end].square().mean().sqrt()),
                    output_change_max=float(change[:,start:end].abs().max()),
                    note='numeric dependence only, not rendering quality attribution or physical unit comparison'))
        assert scene.unchanged()
        del net,payload
        torch.cuda.empty_cache()
    rt.write_csv(ROOT/'analysis_B/rotation_input_dependence.csv',rows)


if __name__=='__main__':
    main()
