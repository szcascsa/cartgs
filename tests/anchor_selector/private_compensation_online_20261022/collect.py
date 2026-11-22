"""真实在线checkpoint的native/Python选点与属性一致性，独立于质量结论。"""
import csv
import json
from pathlib import Path
import re
import subprocess
import sys
import torch

root=Path('/home/prmi/zwj/private-compensation-online-20261022');repo=root/'source';out=root/'validation'
sys.path.insert(0,str(repo/'python'))
from selector import load_selector,load_selector_metadata,mask_inactive_slots,compensated_attributes
from gaussian_model import GaussianModel

torch.set_grad_enabled(False);torch.set_num_threads(2)
runtime=json.loads((out/'training_runtime.json').read_text());checks=[];statistics=[]
for name,info in runtime.items():
    cp=Path(info['checkpoint']);cloud=next((cp/'point_cloud').glob('iteration_*'));destination=out/(name+'_native.pt')
    with (out/(name+'_native.log')).open('w') as log:
        subprocess.run([repo/'bin/anchor_checkpoint_probe',cp/'selector.pt',cloud/'point_cloud.ply',cloud/'selector_metadata.ply',destination,'explicit_blocks'],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    native=torch.jit.load(str(destination),map_location='cuda').state_dict()
    model=load_selector(str(cp/'selector.pt'),'cuda');g=GaussianModel(3);g.load_ply(str(cloud/'point_cloud.ply'))
    meta=load_selector_metadata(str(cloud/'selector_metadata.ply'),'cuda')
    counts=torch.bincount(model.anchors.rows,minlength=len(model.anchors.features));bank=model.gaussian_compensation
    assert torch.equal(counts,model.anchors.member_counts) and counts.min()>=1 and counts.max()<=10
    assert len(torch.unique(bank.gaussian_uids))==len(g.get_xyz)
    for i,ratio in enumerate(native['ratios'].cpu().tolist()):
        p,h=model.forward_selection_features(g.get_xyz,g.get_scaling,g.get_features_dc,g.get_opacity,torch.tensor([[ratio]],device='cuda'))
        p=mask_inactive_slots(model,p,g.get_opacity);mask=meta['cached_uniform']<p;ids=torch.nonzero(mask).flatten()
        assert torch.allclose(p,native['probability_'+str(i)],atol=1e-7,rtol=1e-6)
        assert torch.equal(mask,native['mask_'+str(i)])
        attributes=compensated_attributes(model,g,mask.float(),ratio,h)
        for key,label in [('xyz','xyz'),('scaling','scale'),('rotation','rotation'),('features_dc','dc'),('opacity_logits','opacity_logits')]:
            assert torch.allclose(attributes[key][ids],native['compensated_'+label+'_'+str(i)],atol=2e-6,rtol=2e-5)
        checks.append(dict(model=name,ratio=ratio,selected=len(ids),probability_parity=True,mask_exact=True,all_attributes_parity=True))
    schedule=info['schedule'];stage=info['stages'];anchor=info['anchor']
    clones=re.findall(r'Variable Gaussian Densify\] clones=(\d+) split_parents=(\d+) net_added=(\d+)',(cp.parent.parent/'training.log').read_text())
    statistics.append(dict(model=name,method=info['method'],seed=info['seed'],iteration=anchor['iteration'],gaussians=len(g.get_xyz),anchors=len(counts),
        levels0=int((model.anchors.levels==0).sum()),levels1=int((model.anchors.levels==1).sum()),levels2=int((model.anchors.levels==2).sum()),
        occupancy_mean=float(counts.float().mean()),full10=int((counts==10).sum()),inactive=int(model.anchors.slot_inactive.sum()),
        full_renders=schedule['full_render_count'],lod_renders=schedule['lod_render_count'],
        gaussian_updates=sum(stage[s]['gaussian_updates'] for s in ['online','tail']),
        selector_updates=schedule['selector_online_lod_update_count']+schedule['selector_tail_lod_update_count'],
        compensation_forwards=anchor['compensation_forwards'],private_parameters=sum(x.numel() for x in bank.parameters()),
        clones=sum(int(x[0]) for x in clones),split_parents=sum(int(x[1]) for x in clones),
        wall_seconds=info['wall_seconds'],peak_gpu_mib_sampled=info['peak_gpu_mib_sampled']))
for filename,rows in [('online_model_statistics.csv',statistics),('native_python_checks.csv',checks)]:
    with (out/filename).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'online_correctness.json').write_text(json.dumps(dict(passed=True,runs=len(runtime),native_python_cases=len(checks),all_gaussian_uids_unique=True)))
print('ONLINE_CORRECTNESS_PASSED',flush=True)
