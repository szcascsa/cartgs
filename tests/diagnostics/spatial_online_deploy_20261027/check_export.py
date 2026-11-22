"""真实在线checkpoint的C++/Python概率、选点及补偿属性核对。"""
import json,sys
from pathlib import Path
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/home/prmi/zwj/mycartgs/python')
from selector import load_selector,compensated_attributes,load_selector_metadata
from gaussian_model import GaussianModel
torch.backends.cuda.matmul.allow_tf32=False
cp=Path(json.loads((ROOT/'online_live_fused/completion.json').read_text())['checkpoint'][0]);cloud=next((cp/'point_cloud').glob('iteration_*'))
model=load_selector(str(cp/'selector.pt'),'cuda');g=GaussianModel(3);g.load_ply(str(cloud/'point_cloud.ply'))
meta=load_selector_metadata(str(cloud/'selector_metadata.ply'),'cuda');native=torch.jit.load(str(ROOT/'native_checkpoint_probe.pt')).state_dict();rows=[]
def check(name,a,b):
    d=(a-b).abs();bad=d>1e-6+1e-5*b.abs();rows.append(dict(name=name,max_abs=float(d.max()),rms=float(d.square().mean().sqrt()),bad=int(bad.sum())));assert not bad.any(),name
with torch.no_grad():
    for j,ratio in enumerate(native['ratios'].cpu().tolist()):
        p=model(g.get_xyz,g.get_rotation,g.get_scaling,torch.tensor([[ratio]],device='cuda'),sh_dc=g.get_features_dc,opacity=g.get_opacity)
        inactive=model.anchors.slot_inactive&(g.get_opacity.flatten()<1/255);p=p*(~inactive)
        check(f'{ratio}/probability',p,native[f'probability_{j}'])
        mask=meta['cached_uniform']<p;assert torch.equal(mask,native[f'mask_{j}'])
        ids=mask.nonzero().flatten();a=compensated_attributes(model,g,mask.float(),ratio)
        for name,key in [('xyz','xyz'),('scaling','scale'),('rotation','rotation'),('features_dc','dc'),('opacity_logits','opacity_logits')]:
            check(f'{ratio}/{name}',a[name][ids],native[f'compensated_{key}_{j}'])
pd.DataFrame(rows).to_csv(ROOT/'export_parity.csv',index=False)
(ROOT/'export_parity.json').write_text(json.dumps(dict(pass_=True,rows=len(rows),max_abs=max(r['max_abs'] for r in rows),exact_cached_selection=True),indent=2));print('EXPORT_PARITY_PASS',len(rows),flush=True)
