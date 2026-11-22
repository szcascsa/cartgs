"""固定终点后检查选点变化与激活退化；不用于选择训练参数。"""
from pathlib import Path
import csv,json,sys
import numpy as np,torch
root=Path('/home/prmi/zwj/anchor-si-20261024');sys.path.insert(0,str(root/'source/python'))
from selector import load_selector,load_selector_metadata,mask_inactive_slots
from gaussian_model import GaussianModel
torch.set_grad_enabled(False);torch.set_num_threads(2)
base=Path('/home/prmi/zwj/private-compensation-online-20261022/r3_independent14_seed2026/7465_shutdown/ply')
rows=[];sets={};ratios=[.01,.05,.1,.15]
def inspect(name,cp,selector):
    cloud=next((cp/'point_cloud').glob('iteration_*'));g=GaussianModel(3);g.load_ply(str(cloud/'point_cloud.ply'))
    m=load_selector(str(selector),'cuda');metadata=load_selector_metadata(str(cloud/'selector_metadata.ply'),'cuda')
    for ratio in ratios:
        p,h=m.forward_selection_features(g.get_xyz,g.get_scaling,g.get_features_dc,g.get_opacity,torch.tensor([[ratio]],device='cuda'))
        hidden=m.local_emd[3](m.local_emd[2](h));logits=m.soft_net(hidden)
        eligible=~(m.anchors.slot_inactive.flatten()&(g.get_opacity.flatten()<1/255));effective=mask_inactive_slots(m,p,g.get_opacity)
        margin=((logits[:,1]-logits[:,0]).double()-torch.logit(metadata['cached_uniform'].double())).cpu().numpy();margin[~eligible.cpu().numpy()]=-np.inf
        k=int((metadata['cached_uniform']<effective).sum());order=np.lexsort((np.arange(len(p)),-margin));sets[(name,ratio)]=order
        rows.append(dict(model=name,ratio=ratio,N=len(p),cached_K=k,p_min=float(p.min()),p_max=float(p.max()),p_std=float(p.std(unbiased=False)),prob_saturated_fraction=float(((p<1e-5)|(p>1-1e-5)).float().mean()),prob_exact_extremes_fraction=float(((p==0)|(p==1)).float().mean()),mean_softmax_gate=float((p*(1-p)).mean()),logit_abs_p90=float(torch.quantile((logits[:,1]-logits[:,0]).abs(),.9)),
            shared_row_variation=float((h-h.mean(0)).abs().max()),hidden_zero_row_fraction=float((hidden==0).all(1).float().mean()),hidden_active_fraction=float((hidden>0).float().mean()),
            eligible=int(eligible.sum()),hidden_row_variation=float((hidden-hidden.mean(0)).abs().max())))
for seed in [2026,2027]:
    for arm in ['off','gaussian','anchor']:inspect(f'frozen_{arm}_{seed}',base,root/f'frozen_{arm}_{seed}/selector.pt')
    for arm in ['off','anchor']:
        cp=next((root/f'online_{arm}_{seed}').glob('*_shutdown/ply'));inspect(f'online_{arm}_{seed}',cp,cp/'selector.pt')
with (root/'selection_probe.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
overlap=[];common=json.loads((root/'frozen_evaluation_protocol.json').read_text())['common_K']
for seed in [2026,2027]:
    for arm in ['gaussian','anchor']:
        for ratio in ratios:
            k=common[str(ratio)];a=sets[(f'frozen_off_{seed}',ratio)][:k];b=sets[(f'frozen_{arm}_{seed}',ratio)][:k];shared=len(np.intersect1d(a,b))
            overlap.append(dict(seed=seed,arm=arm,ratio=ratio,K=k,replaced=k-shared,replacement_fraction=(k-shared)/max(k,1)))
with (root/'selection_overlap.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(overlap[0]));w.writeheader();w.writerows(overlap)
print('PROBE_PASS')
