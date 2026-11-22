"""终点反事实诊断；不以non-Bank结果选择模型或训练配置。"""
import argparse,json,types
from pathlib import Path
import torch
from af_runtime import ROOT,Study,RATIOS,TRAIN,evaluate,csv,save,sha,old


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--seed',type=int,default=7200);p.add_argument('--model',default='Head8');p.add_argument('--tier',default='confirm_continue');a=p.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False;study=Study(a.scene)
    directory=ROOT/'results'/a.tier/a.scene/a.model/str(a.seed)
    info=json.loads((directory/'complete.json').read_text());payload=torch.load(info['checkpoint'],map_location='cuda')
    net=study.make(info['config'],a.seed);net.load_state_dict(payload['state_dict']);net.eval()
    seen=torch.zeros(study.scene.N,dtype=torch.bool,device='cuda');signature=torch.zeros(study.M,dtype=torch.long,device='cuda')
    for j,r in enumerate(TRAIN):
        ids=study.fixture.sets[r,'same_k'];seen[ids]=True;signature[study.owners[ids].unique()]+=1<<j
    generator=torch.Generator(device='cuda').manual_seed(9912);mapping=torch.arange(study.M,device='cuda')
    # Match exact occupancy and four-ratio Anchor access pattern; never replace trained features with cold zeros.
    key=study.counts*16+signature
    for value in key.unique():
        ids=(key==value).nonzero().flatten();mapping[ids]=ids[torch.randperm(len(ids),generator=generator,device='cuda')]
    study.shuffle_anchors=mapping
    support=[]
    for r in RATIOS:
        ids=study.fixture.sets[r,'same_k'];owner=study.owners[ids]
        support.append(dict(ratio=r,K=len(ids),gaussian_seen=int(seen[ids].sum()),anchor_seen=int((signature[owner]>0).sum()),
            gaussian_unseen_anchor_seen=int((~seen[ids]&(signature[owner]>0)).sum()),cold_anchor=int((signature[owner]==0).sum()),
            one=int((study.counts[owner]==1).sum()),two_three=int(((study.counts[owner]>=2)&(study.counts[owner]<=3)).sum()),four_ten=int((study.counts[owner]>=4).sum())))
    target=directory/'mechanisms';target.mkdir(exist_ok=True);csv(target/'selection_support.csv',support)
    rows=[]
    for split in ['validation','test']:
        for mode in ['normal','zero','shuffle','no_single','no_small','no_large','no_unseen_member','xyz_feature_off','scale_feature_off','rotation_feature_off','other_feature_off']:
            hook=None;original=None
            if mode=='no_unseen_member':
                original=net.latent
                def filtered(self,ids,ratio):return original(ids,ratio)*seen[ids,None]
                net.latent=types.MethodType(filtered,net)
            if mode.endswith('_feature_off'):
                name=mode.replace('_feature_off','');hook=net.inject[name].register_forward_hook(lambda module,args,out:torch.zeros_like(out))
            base_mode=mode if mode in ['normal','zero','shuffle','no_single','no_small','no_large'] else 'normal'
            data=evaluate(study,net,split,False,base_mode)
            for row in data:row.update(ablation=mode,seed=a.seed,model=a.model)
            rows+=data;csv(target/'counterfactual_per_view.csv',rows)
            if hook is not None:hook.remove()
            if original is not None:net.latent=original
            print('ABLATION',a.scene,a.seed,split,mode,flush=True)
    save(target/'manifest.json',dict(checkpoint=info['checkpoint'],checkpoint_sha=sha(info['checkpoint']),
        shuffle='within exact occupancy and four-ratio Anchor access signature; cold/trained not mixed',
        moved_trained_fraction=float((mapping[signature>0]!=torch.arange(study.M,device='cuda')[signature>0]).float().mean()),
        interpretation='fixed trained base weights; zero/shuffle are counterfactual dependence tests, not retrained controls; group effects nonadditive',
        no_test_tuning=True))
    # Same camera: Anchor-table gradient direction across ratios, restricted to common active rows.
    gradients=[];norms=[]
    for ratio in TRAIN:
        ids=study.fixture.sets[ratio,'same_k'];net.zero_grad(set_to_none=True)
        image,_,_=study.scene.render(ids,net(ratio,ids),study.scene.train[0]);loss=old.rt.photo(image,study.scene.train[0]['gt']);loss.backward()
        grad=net.embedding.grad.detach().clone();gradients.append(grad)
        norms.append(dict(ratio=ratio,norm=float(grad.norm()),nonzero_rows=int((grad.norm(dim=1)>0).sum())))
    pairs=[]
    for i in range(4):
        for j in range(i+1,4):
            first,second=gradients[i],gradients[j];valid=(first.norm(dim=1)>0)&(second.norm(dim=1)>0)
            c=torch.nn.functional.cosine_similarity(first[valid],second[valid],dim=1)
            pairs.append(dict(first=TRAIN[i],second=TRAIN[j],common_anchors=int(valid.sum()),mean_cos=float(c.mean()) if len(c) else None,negative_fraction=float((c<0).float().mean()) if len(c) else None))
    save(target/'ratio_gradient.json',dict(norms=norms,pairs=pairs,limit='one fixed Bank camera, final checkpoint; cosine is not proof of quality harm'))
    # Fixed identities isolate neural continuity from discrete Selector membership changes.
    ids=study.fixture.sets[.01,'same_k'];continuity=[];previous=None
    with torch.no_grad():
        for ratio in torch.linspace(.01,.15,141).tolist():
            value=net(ratio,ids);assert value.isfinite().all()
            continuity.append(dict(ratio=ratio,rows=len(ids),max_abs=float(value.abs().max()),xyz_anchor_units_p90=float((value[:,:3]/study.anchor_scale[ids]).norm(dim=1).quantile(.9)),adjacent_max=float((value-previous).abs().max()) if previous is not None else 0.))
            previous=value
    csv(target/'dense_ratio_continuity.csv',continuity)
if __name__=='__main__':main()
