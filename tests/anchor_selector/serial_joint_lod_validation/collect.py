"""独立复算串行 Full/LoD、真实训练计数与质量，保护既有实验和 production。"""
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
from plyfile import PlyData

root=Path('/home/prmi/zwj/serial-joint-lod-20261021');repo=root/'source';out=root/'validation'
previous=Path('/home/prmi/zwj/anchor-slot-deactivation-20261021')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((root/'source_manifest.json').read_text())
assert all(sha(repo/p)==h for p,h in manifest.items())
original=json.loads((previous/'source_manifest.json').read_text())
assert all(sha(previous/'source'/p)==h for p,h in original.items())
production=json.loads(Path('/home/prmi/zwj/octree-explicit-20261021/validation/source_manifest.json').read_text())
assert all(sha(Path('/home/prmi/zwj/mycartgs')/p)==h for p,h in production.items())
plan=json.loads((out/'evaluation_plan.json').read_text())
assert all(sha(p)==h for p,h in plan['model_sha256'].items())
gates=json.loads((out/'correctness.json').read_text());assert len(gates)==8 and all(v['passed'] for v in gates.values())
run=root/'joint_seed2026';checkpoint=run/'10000_shutdown/ply';cloud=checkpoint/'point_cloud/iteration_10000'
trace=[{k:int(v) for k,v in row.items()} for row in csv.DictReader((run/'serial_full_lod_trace.csv').open())]
assert len(trace)==10000 and [r['mapping'] for r in trace]==list(range(1,10001))
for row in trace:
    assert row['full']==row['lod']==row['full_gaussian_update']==row['selector_update']==1
    assert row['lod_gaussian_update']==1 and row['ratio_index']==(row['mapping']-1)%4
state=json.loads((checkpoint/'ultrafast_rotation_state.json').read_text())
assert state['serial_full_lod'] and not state['serial_lod_detach_master'] and state['post_topology_full_steps']==0
assert state['full_render_count']==state['lod_render_count']==state['ultrafast_total_normal_steps']==10000
assert state['recovery_windows']==state['recovery_steps']==state['recovery_remaining']==0 and state['balanced']
stages=json.loads((checkpoint/'mapping_stage_state.json').read_text())
assert sum(stages[p]['gaussian_updates'] for p in ['online','tail'])==20000
assert sum(stages[p]['lod_nonfinite'] for p in ['online','tail'])==0
assert [sum(stages[p]['ratio_counts'][i] for p in ['online','tail']) for i in range(4)]==[2500]*4

def write(name,rows):
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
sys.path.insert(0,str(repo/'python'))
from selector import load_selector,load_selector_metadata,build_selector_mask
from gaussian_model import GaussianModel
model=load_selector(str(checkpoint/'selector.pt'),'cuda')
g=GaussianModel(3);g.load_ply(str(cloud/'point_cloud.ply'))
meta=load_selector_metadata(str(cloud/'selector_metadata.ply'),'cuda')
with (out/'real_checkpoint_native.log').open('w') as f:
    subprocess.run([str(repo/'bin/anchor_checkpoint_probe'),str(checkpoint/'selector.pt'),str(cloud/'point_cloud.ply'),
        str(cloud/'selector_metadata.ply'),str(out/'real_checkpoint_native.pt'),'explicit_blocks'],stdout=f,stderr=subprocess.STDOUT,check=True)
fixture=torch.jit.load(str(out/'real_checkpoint_native.pt'),map_location='cuda').state_dict()
config=dict(drift_temperature=.1,beta_fast=.6,beta_slow=.15,compatibility_mature_threshold=.8,entry_maturity=0)
for i,r in enumerate([.01,.05,.1,.15]):
    mask,_=build_selector_mask(model,g,meta,r,config,1.,False)
    assert torch.equal(mask.bool(),fixture['mask_'+str(i)])
    ids=np.load(root/'quality/joint/cached_bernoulli'/str(int(r*100))/'selected_rows.npz')['rows']
    assert np.array_equal(ids,torch.nonzero(mask).flatten().cpu().numpy())
stats=[]
for name,result in [('baseline',previous/'enabled_seed2026'),('detached',Path('/home/prmi/zwj/serial-full-lod-20261021/serial_seed2026')),('joint',run)]:
    cp=result/'10000_shutdown/ply';pc=cp/'point_cloud/iteration_10000'
    ply=PlyData.read(pc/'point_cloud.ply')['vertex'].data
    alpha=1/(1+np.exp(-np.clip(ply['opacity'].astype(np.float64),-700,700)))
    network=load_selector(str(cp/'selector.pt'),'cpu')
    schedule=json.loads((cp/'ultrafast_rotation_state.json').read_text())
    stages=json.loads((cp/'mapping_stage_state.json').read_text())
    if name=='baseline':
        (out/'baseline_ultrafast_rotation_state.json').write_bytes((cp/'ultrafast_rotation_state.json').read_bytes())
        (out/'baseline_mapping_stage_state.json').write_bytes((cp/'mapping_stage_state.json').read_bytes())
    text=(result/'training.log').read_text()
    a=alpha.reshape(-1,10);low=a<1/255
    stats.append(dict(method=name,anchors=len(a),gaussians=len(alpha),full_renders=schedule['full_render_count'],
        lod_renders=schedule['lod_render_count'],online_end_mapping=schedule['online_end_mapping_iteration'],
        gaussian_updates=stages['online']['gaussian_updates']+stages['tail']['gaussian_updates'],
        selector_updates=schedule['selector_online_lod_update_count']+schedule['selector_tail_lod_update_count'],
        tf_updates=schedule['tf_online_lod_update_count']+schedule['tf_tail_lod_update_count'],
        below_cutoff=int(low.sum()),below_cutoff_fraction=float(low.mean()),below_002=int((a<.02).sum()),
        anchors_ge8_below_cutoff=int((low.sum(1)>=8).sum()),anchors_all10_below_cutoff=int(low.all(1).sum()),
        opacity_mean=float(alpha.mean()),opacity_median=float(np.median(alpha)),
        inactive=int(network.anchors.slot_inactive.sum()),
        created_anchors=sum(map(int,re.findall(r'Explicit Anchor Insert\] added=(\d+)',text))),
        pruned_anchors=sum(map(int,re.findall(r'Anchor Opacity Prune\].*?removed=(\d+)',text)))))
    assert stats[-1]['created_anchors']-stats[-1]['pruned_anchors']==len(a)
write('optimization_diagnostics.csv',stats)
reference=Path('/home/prmi/zwj/serial-full-lod-20261021')
reference_source=json.loads((reference/'source_manifest.json').read_text())
assert all(sha(reference/'source'/p)==h for p,h in reference_source.items())
diagnostics=dict(correctness_gates=len(gates),trace_rows=len(trace),
    full_render_mappings=10000,lod_render_mappings=10000,lod_master_updates=10000,
    full100_disabled=True,ratio_counts=[2500]*4,original_source_unchanged=True,
    detached_source_unchanged=True,production_unchanged=True,
    real_native_python_export_masks_equal=True,quality_models_sha_unchanged=True,
    mapping_budget=10000,training_runs=1,optimization_statistics=stats)
(out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
files=list(out.glob('*.json'))+list(out.glob('*.csv'))+list(out.glob('*.log'))+list(out.glob('*.pt'))
files += [p for p in root.glob('*.py')]+[root/'pipeline.log',run/'training.log',run/'input_selector.yaml',
    run/'serial_full_lod_trace.csv',run/'CameraTrajectory_TUM.txt']
files+=list((run/'anchor_pruning').glob('*'))
files+=[checkpoint/'selector.pt',checkpoint/'selector_config.yaml',checkpoint/'transform_field.pt',checkpoint/'transform_state.json',
    checkpoint/'ultrafast_rotation_state.json',checkpoint/'mapping_stage_state.json',cloud/'point_cloud.ply',
    cloud/'selector_metadata.ply',cloud/'anchor_pruning_state.pt',cloud/'anchor_selector_state.json']
for result in [previous/'enabled_seed2026',run]:
    for p in (result/'10000_shutdown/ply').glob('*state.json'):
        if result==run:files.append(p)
for folder in (root/'quality').glob('*/*/*'):
    files += [p for p in folder.iterdir() if p.suffix in ['.txt','.json','.log','.npz']]
files=sorted(set(p for p in files if p.is_file()))
record=dict(source_sha256=manifest,original_source_unchanged=True,production_unchanged=True,
    models_unchanged=True,file_sha256={str(p.relative_to(root)):sha(p) for p in files})
(root/'delivery_verification.json').write_text(json.dumps(record,indent=2));files+=[root/'delivery_verification.json']
with tarfile.open(root/'evidence.tar.gz','w:gz') as t:
    for p in files:t.add(p,arcname=str(p.relative_to(root)))
print(json.dumps(dict(archive_sha256=sha(root/'evidence.tar.gz'),files=len(files),diagnostics=diagnostics),indent=2),flush=True)
