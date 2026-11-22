"""复核原 joint、冻结模型与原生选择一致性，归档质量原始数据。"""
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

root = Path('/home/prmi/zwj/variable-anchor-joint-quality-20261021')
repo = Path('/home/prmi/zwj/variable-anchor-20261021/source')
out = root/'validation'
sys.path.insert(0, str(repo/'python'))
from selector import load_selector, load_selector_metadata, build_selector_mask
from gaussian_model import GaussianModel
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan = json.loads((out/'evaluation_plan.json').read_text())
assert all(sha(p) == h for p,h in plan['model_sha256'].items())
protocol = json.loads((out/'training_protocol.json').read_text())
assert all(sha(repo/p) == h for p,h in protocol['source_sha256'].items())
assert json.loads((out/'quality_complete.json').read_text()) == dict(completed=True,model_sha_unchanged=True,arms=18,per_view_rows=1152,paired_rows=36)
jobs = dict(fixed=Path('/home/prmi/zwj/anchor-slot-deactivation-20261021/enabled_seed2026'), variable=root/'variable_seed2026')
rows = []
for name, run in jobs.items():
    cp = run/'10000_shutdown/ply'
    cloud = cp/'point_cloud/iteration_10000'
    schedule = json.loads((cp/'ultrafast_rotation_state.json').read_text())
    stages = json.loads((cp/'mapping_stage_state.json').read_text())
    assert not schedule.get('serial_full_lod',False) and schedule.get('post_topology_full_steps',0) == 0
    assert schedule['ultrafast_total_normal_steps'] == 10000 and schedule['balanced']
    assert sum(stages[p]['lod_nonfinite'] for p in ['online','tail']) == 0
    assert sum(stages[p]['gaussian_updates'] for p in ['online','tail']) == schedule['full_render_count']+schedule['lod_render_count']
    model = load_selector(str(cp/'selector.pt'),'cuda')
    owners = model.anchors.rows
    counts = torch.bincount(owners,minlength=len(model.anchors.features))
    assert ((counts>=1)&(counts<=10)).all()
    if name == 'fixed':
        assert not hasattr(model.anchors,'member_counts') and (counts == 10).all()
    else:
        assert torch.equal(counts,model.anchors.member_counts)
        events = re.findall(r'Alternating Topology\] round=(\d+) phase=(Gaussian|Anchor) mapping=(\d+)',(run/'training.log').read_text())
        assert [int(e[0]) for e in events] == list(range(len(events)))
        assert all(p == ('Gaussian' if int(i)%2 == 0 else 'Anchor') for i,p,_ in events)
        assert len(events) == int(model.anchors.topology_round)
    g = GaussianModel(3)
    g.load_ply(str(cloud/'point_cloud.ply'))
    metadata = load_selector_metadata(str(cloud/'selector_metadata.ply'),'cuda')
    assert (metadata['maturity'] == 1).all() and len(g.get_xyz) == plan['population'][name] == len(owners)
    with (out/(name+'_native.log')).open('w') as log:
        subprocess.run([str(repo/'bin/anchor_checkpoint_probe'),str(cp/'selector.pt'),str(cloud/'point_cloud.ply'),
                        str(cloud/'selector_metadata.ply'),str(out/(name+'_native.pt')),'explicit_blocks'],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    fixture = torch.jit.load(str(out/(name+'_native.pt')),map_location='cuda').state_dict()
    config = dict(drift_temperature=.1,beta_fast=.6,beta_slow=.15,compatibility_mature_threshold=.8,entry_maturity=0)
    for i,r in enumerate([.01,.05,.1,.15]):
        mask,_ = build_selector_mask(model,g,metadata,r,config,1.,False)
        assert torch.equal(mask.bool(),fixture['mask_'+str(i)])
        saved = np.load(root/'quality'/name/'cached_bernoulli'/str(int(r*100))/'selected_rows.npz')['rows']
        assert np.array_equal(saved,torch.nonzero(mask).flatten().cpu().numpy())
    alpha = g.get_opacity.flatten()
    rows.append(dict(method=name,gaussians=len(owners),anchors=len(counts),
                     occupancy_mean=float(counts.float().mean()),
                     below_cutoff=int((alpha<1/255).sum()),below_002=int((alpha<.02).sum()),
                     inactive=int(model.anchors.slot_inactive.sum()),
                     full_renders=schedule['full_render_count'],lod_renders=schedule['lod_render_count'],
                     gaussian_updates=sum(stages[p]['gaussian_updates'] for p in ['online','tail']),
                     selector_updates=schedule['selector_online_lod_update_count']+schedule['selector_tail_lod_update_count'],
                     tf_updates=schedule['tf_online_lod_update_count']+schedule['tf_tail_lod_update_count'],
                     online_end_mapping=schedule['online_end_mapping_iteration'],
                     native_forward_ms=float(fixture['native_forward_ms'].mean()),
                     selector_feature_bytes=model.anchors.features.numel()*model.anchors.features.element_size(),
                     occupancy_histogram_0_to_10=torch.bincount(counts,minlength=11).cpu().tolist(),
                     anchors_by_level=torch.bincount(model.anchors.levels,minlength=3).cpu().tolist()))
    for file in ['ultrafast_rotation_state.json','mapping_stage_state.json']:
        (out/(name+'_'+file)).write_bytes((cp/file).read_bytes())
with (out/'model_diagnostics.csv').open('w',newline='') as stream:
    writer = csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
result = dict(passed=True,serial_joint=False,baseline_reused=str(jobs['fixed']),
              excluded_duplicate_fixed=str(root/'fixed_seed2026'),
              model_sha_unchanged=True,source_sha_verified=len(protocol['source_sha256']),
              native_python_export_masks_exact=True,common_frames=64,common_k=plan['common_k'],
              arms=18,per_view_rows=1152,paired_rows=36,models=rows,
              interpretation='one scene and one seed; population, optimization trajectory, TF and poses differ')
(out/'audit_summary.json').write_text(json.dumps(result,indent=2))
files = list(out.glob('*.json'))+list(out.glob('*.csv'))+list(out.glob('*.log'))+list(out.glob('*.pt'))+list(root.glob('*.py'))
files += [root/'training_pipeline.log',root/'evaluation_pipeline.log',jobs['variable']/'training.log',jobs['variable']/'input_selector.yaml',jobs['variable']/'CameraTrajectory_TUM.txt']
cp = jobs['variable']/'10000_shutdown/ply';cloud = cp/'point_cloud/iteration_10000'
files += [cp/'selector.pt',cp/'selector_config.yaml',cp/'transform_field.pt',cp/'transform_state.json',cloud/'point_cloud.ply',cloud/'selector_metadata.ply',cloud/'anchor_pruning_state.pt',cloud/'anchor_selector_state.json']
for folder in (root/'quality').glob('*/*/*'):
    files += [p for p in folder.iterdir() if p.suffix in ['.txt','.json','.log','.npz']]
files = sorted(set(p for p in files if p.is_file()))
manifest = {str(p.relative_to(root)):sha(p) for p in files}
(root/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(root/'evidence.tar.gz','w:gz') as archive:
    for p in files+[root/'evidence_manifest.json']:
        archive.add(p,arcname=str(p.relative_to(root)))
print(json.dumps(dict(files=len(files),archive_sha256=sha(root/'evidence.tar.gz'),audit=result),indent=2))
