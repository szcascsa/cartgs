"""核验 topology、正式选择与渲染属性；收集三模型质量与人口证据。"""
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

root=Path('/home/prmi/zwj/anchor-only-prune-20261021');repo=root/'source';out=root/'validation'
baseline=Path('/home/prmi/zwj/shared-selector-appearance-20261021')
sys.path.insert(0,str(repo/'python'))
from selector import load_selector,load_selector_metadata,build_selector_mask,compensated_attributes
from gaussian_model import GaussianModel
from renderer import prepare_render_data
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=json.loads((out/'evaluation_plan.json').read_text())
assert json.loads((out/'quality_complete.json').read_text())['completed']
assert all(sha(p)==h for p,h in plan['model_sha256'].items())
source=json.loads((root/'source_manifest.json').read_text());assert all(sha(repo/p)==h for p,h in source.items())
jobs=dict(baseline=baseline/'candidate_seed2026',anchor_only=root/'anchor_only_seed2026',anchor_only_full10=root/'anchor_only_full10_seed2026')
config=dict(drift_temperature=.1,beta_fast=.6,beta_slow=.15,compatibility_mature_threshold=.8,entry_maturity=0)
diagnostics=[];selected_stats=[];growth_stats=[]
for name,run in jobs.items():
    cp=run/'10000_shutdown/ply';cloud=cp/'point_cloud/iteration_10000'
    schedule=json.loads((cp/'ultrafast_rotation_state.json').read_text())
    stages=json.loads((cp/'mapping_stage_state.json').read_text());anchors=json.loads((cloud/'anchor_selector_state.json').read_text())
    assert not schedule['serial_full_lod'] and schedule['post_topology_full_steps']==0 and schedule['balanced']
    assert schedule['ultrafast_total_normal_steps']==10000 and sum(stages[x]['lod_nonfinite'] for x in ['online','tail'])==0
    assert not (cp/'transform_field.pt').exists() and not (cp/'transform_state.json').exists()
    assert schedule['tf_online_lod_update_count']==schedule['tf_tail_lod_update_count']==0
    model=load_selector(str(cp/'selector.pt'),'cuda');g=GaussianModel(3);g.load_ply(str(cloud/'point_cloud.ply'))
    assert all(torch.isfinite(p).all() for p in model.parameters()) and model.compensation[-1].out_features==14
    owners=model.anchors.rows;counts=torch.bincount(owners,minlength=len(model.anchors.features))
    assert torch.equal(counts,model.anchors.member_counts) and ((counts>0)&(counts<=10)).all()
    metadata=load_selector_metadata(str(cloud/'selector_metadata.ply'),'cuda');assert (metadata['maturity']==1).all()
    log=(run/'training.log').read_text()
    rounds=re.findall(r'Alternating Topology\] round=(\d+) phase=(Gaussian|Anchor) mapping=(\d+)',log)
    assert [int(i) for i,_,_ in rounds]==list(range(len(rounds)))
    assert all(p==('Gaussian' if int(i)%2==0 else 'Anchor') for i,p,_ in rounds)
    assert len(rounds)==int(model.anchors.topology_round)
    prune_logs=re.findall(r'Variable Gaussian Prune\] enabled=(\d+) removed=(\d+)',log)
    growth=[]
    for line in log.splitlines():
        if '[Explicit Anchor Growing]' in line:
            fields=dict(re.findall(r'(\w+)=([^\s]+)',line))
            row=dict(method=name,event=len(growth),**fields);growth.append(row)
    growth_stats+=growth
    if name!='baseline':
        assert anchors['anchor_only_pruning'] and anchors['gaussian_pruned_members']==0
        assert len(prune_logs)==sum(p=='Gaussian' for _,p,_ in rounds) and all(x==('0','0') for x in prune_logs)
        assert bool(anchors['anchor_refine_requires_full'])==(name=='anchor_only_full10')
        assert all(int(v['refine_requires_full'])==(name=='anchor_only_full10') for v in growth)
        if name=='anchor_only_full10':assert all(int(v['fine_partial_candidates'])==0 for v in growth)
    with (out/(name+'_native.log')).open('w') as stream:
        subprocess.run([str(repo/'bin/anchor_checkpoint_probe'),str(cp/'selector.pt'),str(cloud/'point_cloud.ply'),
            str(cloud/'selector_metadata.ply'),str(out/(name+'_native.pt')),'explicit_blocks'],stdout=stream,stderr=subprocess.STDOUT,check=True)
    native=torch.jit.load(str(out/(name+'_native.pt')),map_location='cuda').state_dict()
    for i,r in enumerate([.01,.05,.1,.15]):
        mask,_=build_selector_mask(model,g,metadata,r,config,1.,False);ids=torch.nonzero(mask).flatten()
        assert torch.equal(mask.bool(),native['mask_'+str(i)])
        folder=root/'quality'/name/'cached_bernoulli'/str(int(r*100))
        assert np.array_equal(np.load(folder/'selected_rows.npz')['rows'],ids.cpu().numpy())
        with torch.no_grad():attributes=compensated_attributes(model,g,mask,r)
        for attr,native_name in [('xyz','xyz'),('scaling','scale'),('rotation','rotation')]:
            assert torch.allclose(attributes[attr][ids],native['compensated_'+native_name+'_'+str(i)],atol=2e-6,rtol=2e-5)
        assert torch.allclose(attributes['features_dc'][ids],native['compensated_dc_'+str(i)],atol=2e-6,rtol=2e-5)
        assert torch.allclose(attributes['opacity_logits'][ids],native['compensated_opacity_logits_'+str(i)],atol=2e-6,rtol=2e-5)
        if name!='baseline':
            exported=GaussianModel(3);exported.load_ply(str(folder/'point_cloud.ply'))
            assert torch.equal(exported._opacity,attributes['opacity_logits'][ids])
            assert torch.equal(exported.get_features_dc,attributes['features_dc'][ids])
            assert torch.equal(exported.get_features_rest,g.get_features_rest[ids])
            prepared=prepare_render_data(g,mask,attributes)['inputs']
            assert torch.equal(prepared['opacities'],exported.get_opacity) and torch.equal(prepared['dc'],exported.get_features_dc)
        raw=g.get_opacity[ids].flatten();alpha=attributes['opacity_logits'][ids].sigmoid().flatten()
        selected_stats.append(dict(method=name,ratio=r,selected=len(ids),raw_below_cutoff=int((raw<1/255).sum()),
            raw_below_002=int((raw<.02).sum()),compensated_below_cutoff=int((alpha<1/255).sum()),
            raw_opacity_mean=float(raw.mean()),compensated_opacity_mean=float(alpha.mean())))
    total=lambda key:sum(int(v.get(key,0)) for v in growth)
    clone_split=re.findall(r'Variable Gaussian Densify\] clones=(\d+) split_parents=(\d+) net_added=(\d+)',log)
    alpha=g.get_opacity.flatten()
    row=dict(method=name,gaussians=len(g.get_xyz),anchors=len(counts),occupancy_mean=float(counts.float().mean()),
        full_anchors=int((counts==10).sum()),occupancy_histogram=torch.bincount(counts,minlength=11).tolist(),
        anchors_by_level=torch.bincount(model.anchors.levels,minlength=3).tolist(),
        below_cutoff=int((alpha<1/255).sum()),below_002=int((alpha<.02).sum()),inactive=int(model.anchors.slot_inactive.sum()),
        full_renders=schedule['full_render_count'],lod_renders=schedule['lod_render_count'],
        gaussian_updates=sum(stages[x]['gaussian_updates'] for x in ['online','tail']),
        selector_updates=schedule['selector_online_lod_update_count']+schedule['selector_tail_lod_update_count'],
        online_end_mapping=schedule['online_end_mapping_iteration'],compensation_forwards=anchors['compensation_forwards'],
        cd_compensation_steps=anchors['cd_compensation_steps'],topology_rounds=len(rounds),
        gaussian_pruned_members=anchors.get('gaussian_pruned_members'),anchor_pruned_members=anchors.get('anchor_pruned_members'),
        clones=sum(int(v[0]) for v in clone_split),split_parents=sum(int(v[1]) for v in clone_split),
        same_candidates=total('same_candidates'),fine_candidates=total('fine_candidates'),
        fine_full_candidates=total('fine_full_candidates'),fine_partial_candidates=total('fine_partial_candidates'),
        fine_blocked_not_full=total('fine_blocked_not_full'),new_anchors_from_growing=total('new_anchors'),
        selector_parameters=sum(p.numel() for key,p in model.named_parameters() if key!='anchors.features'),
        anchor_feature_values=model.anchors.features.numel(),native_forward_ms=float(native['native_forward_ms'].mean()),
        gaussian_parameter_bytes=sum(v.numel()*v.element_size() for v in
            [g._xyz,g._scaling,g._rotation,g._opacity,g._features_dc,g._features_rest]))
    assert row['gaussian_updates']==row['full_renders']+row['lod_renders'];diagnostics.append(row)
    for file in ['ultrafast_rotation_state.json','mapping_stage_state.json']:(out/(name+'_'+file)).write_bytes((cp/file).read_bytes())
def write(name,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with (out/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
write('selected_opacity.csv',selected_stats);write('growth_events.csv',growth_stats)
(out/'model_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
assert all(sha(p)==h for p,h in plan['model_sha256'].items())
result=dict(passed=True,source_sha_verified=len(source),model_sha_unchanged=True,models=diagnostics,
    native_python_masks_exact=True,native_python_ply_appearance=True,renderer_matches_ply=True,
    gaussian_quality_pruning_disabled=True,fine_gate_next_level_only=True,tf_disabled=True,
    limitations='one scene/seed; maps, poses, budgets and online update counts differ; split-parent removal retained')
(out/'audit_summary.json').write_text(json.dumps(result,indent=2))
files=list(out.glob('*.*'))+list(root.glob('*.py'))+list(root.glob('*pipeline.log'))+[root/'source_manifest.json']
for method in ['anchor_only','anchor_only_full10']:
    run=jobs[method];cp=run/'10000_shutdown/ply';cloud=cp/'point_cloud/iteration_10000'
    files+=[run/'training.log',run/'input_selector.yaml',run/'CameraTrajectory_TUM.txt',cp/'selector.pt',cp/'selector_config.yaml',
        cloud/'point_cloud.ply',cloud/'selector_metadata.ply',cloud/'anchor_selector_state.json',cloud/'anchor_pruning_state.pt']
    files+=list((run/'anchor_pruning').glob('*.csv'))
for folder in (root/'quality').glob('*/*/*'):
    files+=[p for p in folder.iterdir() if p.suffix in ['.txt','.json','.log','.npz','.ply']]
files=sorted(set(p for p in files if p.is_file()))
(root/'evidence_manifest.json').write_text(json.dumps({str(p.relative_to(root)):sha(p) for p in files},indent=2))
with tarfile.open(root/'evidence.tar.gz','w:gz') as archive:
    for p in files+[root/'evidence_manifest.json']:archive.add(p,arcname=str(p.relative_to(root)))
(root/'archive_info.json').write_text(json.dumps(dict(sha256=sha(root/'evidence.tar.gz'),bytes=(root/'evidence.tar.gz').stat().st_size)))
print('COLLECTED',len(files),flush=True)
