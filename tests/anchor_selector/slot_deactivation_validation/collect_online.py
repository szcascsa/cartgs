"""训练/质量独立复核，保留原模型与 production，打包受控证据。"""
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import numpy as np
import torch

root=Path('/home/prmi/zwj/anchor-slot-deactivation-20261021');repo=root/'source';out=root/'validation'
old=Path('/home/prmi/zwj/octree-anchor-pruning-20261021')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=json.loads((root/'source_manifest.json').read_text())
assert all(sha(repo/p)==h for p,h in source.items())
baseline_source=json.loads((old/'validation/source_manifest.json').read_text())
assert all(sha(old/'source'/p)==h for p,h in baseline_source.items())
production=json.loads(Path('/home/prmi/zwj/octree-explicit-20261021/validation/source_manifest.json').read_text())
assert all(sha(Path('/home/prmi/zwj/mycartgs')/p)==h for p,h in production.items())
plan=json.loads((out/'evaluation_plan.json').read_text())
assert all(sha(p)==h for p,h in plan['model_sha256'].items())
gates=json.loads((out/'correctness.json').read_text())
assert len(gates)==15 and all(v['passed'] for v in gates.values())
sys.path.insert(0,str(repo/'python'))
from selector import load_selector,load_selector_metadata,build_selector_mask
from gaussian_model import GaussianModel
run=root/'enabled_seed2026';checkpoint=run/'10000_shutdown/ply';cloud=checkpoint/'point_cloud/iteration_10000'
model=load_selector(str(checkpoint/'selector.pt'),'cuda')
g=GaussianModel(3);g.load_ply(str(cloud/'point_cloud.ply'))
meta=load_selector_metadata(str(cloud/'selector_metadata.ply'),'cuda')
assert g.get_xyz.shape[0]==10*len(model.anchors.features)
inactive=model.anchors.slot_inactive.flatten()
assert not (inactive & (g.get_opacity.flatten()>=1/255)).any()
assert not model.anchors.slot_inactive.requires_grad
with (out/'real_checkpoint_native.log').open('w') as f:
    subprocess.run([str(repo/'bin/anchor_checkpoint_probe'),str(checkpoint/'selector.pt'),str(cloud/'point_cloud.ply'),
        str(cloud/'selector_metadata.ply'),str(out/'real_checkpoint_native.pt'),'explicit_blocks'],stdout=f,stderr=subprocess.STDOUT,check=True)
fixture=torch.jit.load(str(out/'real_checkpoint_native.pt'),map_location='cuda').state_dict()
config=dict(drift_temperature=.1,beta_fast=.6,beta_slow=.15,compatibility_mature_threshold=.8,entry_maturity=0)
for i,r in enumerate([.01,.05,.1,.15]):
    mask,stats=build_selector_mask(model,g,meta,r,config,1.,False)
    assert torch.equal(mask.bool(),fixture['mask_'+str(i)])
    assert not (mask.bool()&inactive).any()
    ids=np.load(root/'quality/enabled/cached_bernoulli'/str(int(r*100))/'selected_rows.npz')['rows']
    assert np.array_equal(ids,torch.nonzero(mask).flatten().cpu().numpy())
window_rows=[]
for p in sorted((run/'anchor_pruning').glob('window_*.pt'),key=lambda p:int(p.stem.split('_')[-1])):
    d=torch.jit.load(str(p),map_location='cpu').state_dict()
    enough=d['anchor_visits']>.8*int(d['check_interval'])
    deleted=enough&(d['opacity_accum']<.005*d['anchor_visits'])
    assert torch.allclose(d['slot_opacity_accum'].sum(1),d['opacity_accum'],atol=1e-8,rtol=1e-12)
    assert not (d['slot_inactive'] & (d['slot_current_opacity']>=1/255)).any()
    window_rows.append(dict(mapping=int(d['mapping_iteration']),anchors=len(enough),population=10*len(enough),
        sufficient=int(enough.sum()),deleted_anchors=int(deleted.sum()),inactive_slots=int(d['slot_inactive'].sum())))
with (out/'activity_windows.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(window_rows[0]));w.writeheader();w.writerows(window_rows)
text=(run/'training.log').read_text();assert '10000 iterations completed.' in text
created=sum(map(int,re.findall(r'Explicit Anchor Insert\] added=(\d+)',text)))
deleted=sum(map(int,re.findall(r'Anchor Opacity Prune\].*?removed=(\d+)',text)))
assert created-deleted==len(model.anchors.features)
summary=dict(correctness_gates=len(gates),real_native_python_export_masks_equal=True,original_source_unchanged=True,
    production_unchanged=True,models_unchanged_during_evaluation=True,training_runs=1,
    created_anchors=created,pruned_anchors=deleted,final_anchors=len(model.anchors.features),
    final_gaussians=len(inactive),inactive_slots=int(inactive.sum()),
    current_below_cutoff=int((g.get_opacity<1/255).sum()),
    unmasked_current_below_cutoff=int(((g.get_opacity.flatten()<1/255)&~inactive).sum()),
    levels={str(l):int((model.anchors.levels==l).sum()) for l in [0,1,2]},
    activity_buffer_bytes=sum(b.numel()*b.element_size() for b in [model.anchors.slot_inactive,model.anchors.slot_recovered]),
    native_forward_ms=fixture['native_forward_ms'].cpu().tolist(),
    no_online_quality_tuning=True,source_sha256=source)
(out/'online_diagnostics.json').write_text(json.dumps(summary,indent=2))
files=list(out.glob('*.json'))+list(out.glob('*.csv'))+list(out.glob('*.log'))+list(out.glob('*.pt'))
for device in ['cpu','cuda']:files+=list((out/device).glob('*.pt'))
files+=list((run/'anchor_pruning').glob('*'))
files+=[run/'training.log',run/'input_selector.yaml',run/'CameraTrajectory_TUM.txt',
    checkpoint/'selector.pt',checkpoint/'selector_config.yaml',checkpoint/'transform_field.pt',checkpoint/'transform_state.json',
    cloud/'point_cloud.ply',cloud/'selector_metadata.ply',cloud/'anchor_pruning_state.pt',cloud/'anchor_selector_state.json']
for folder in (root/'quality').glob('*/*/*'):
    files += [p for p in folder.iterdir() if p.suffix in ['.txt','.json','.log','.npz']]
files += [p for p in root.glob('*.py')]+[root/'online.log',root/'online_retry.log',root/'online_final.log']
files=sorted(set(p for p in files if p.is_file()))
record=dict(source_sha_matched=True,production_unchanged=True,baseline_source_unchanged=True,
    baseline_models_unchanged=True,file_sha256={str(p.relative_to(root)):sha(p) for p in files})
(root/'online_delivery_verification.json').write_text(json.dumps(record,indent=2));files+=[root/'online_delivery_verification.json']
with tarfile.open(root/'online_evidence.tar.gz','w:gz') as t:
    for p in files:t.add(p,arcname=str(p.relative_to(root)))
print(json.dumps(dict(archive_sha256=sha(root/'online_evidence.tar.gz'),files=len(files),summary=summary),indent=2),flush=True)
