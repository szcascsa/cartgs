"""复核新旧模型、原生/Python/PLY 外观属性、同集合消融和冻结源文件。"""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import numpy as np
import torch

root=Path('/home/prmi/zwj/shared-selector-appearance-20261021');repo=root/'source';out=root/'validation'
old=Path('/home/prmi/zwj/shared-selector-compensation-20261021')
sys.path.insert(0,str(repo/'python'))
from selector import load_selector,load_selector_metadata,build_selector_mask,compensated_attributes
from gaussian_model import GaussianModel
from renderer import prepare_render_data
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=json.loads((out/'evaluation_plan.json').read_text())
assert json.loads((out/'quality_complete.json').read_text())['completed']
assert all(sha(p)==h for p,h in plan['model_sha256'].items())
source=json.loads((root/'source_manifest.json').read_text());assert all(sha(repo/p)==h for p,h in source.items())
jobs=dict(geometry_v1=old/'candidate_seed2026',appearance_v2=root/'candidate_seed2026')
config=dict(drift_temperature=.1,beta_fast=.6,beta_slow=.15,compatibility_mature_threshold=.8,entry_maturity=0)
diagnostics=[];statistics=[]
for name,run in jobs.items():
    cp=run/'10000_shutdown/ply';cloud=cp/'point_cloud/iteration_10000'
    schedule=json.loads((cp/'ultrafast_rotation_state.json').read_text())
    stages=json.loads((cp/'mapping_stage_state.json').read_text());anchors=json.loads((cloud/'anchor_selector_state.json').read_text())
    assert not schedule['serial_full_lod'] and schedule['post_topology_full_steps']==0 and schedule['balanced']
    assert schedule['ultrafast_total_normal_steps']==10000 and sum(stages[x]['lod_nonfinite'] for x in ['online','tail'])==0
    assert not (cp/'transform_field.pt').exists() and not (cp/'transform_state.json').exists()
    assert schedule['tf_online_lod_update_count']==schedule['tf_tail_lod_update_count']==0
    model=load_selector(str(cp/'selector.pt'),'cuda');g=GaussianModel(3);g.load_ply(str(cloud/'point_cloud.ply'))
    assert all(torch.isfinite(p).all() for p in model.parameters())
    width=model.compensation[-1].out_features;assert width==(14 if name=='appearance_v2' else 10)
    counts=torch.bincount(model.anchors.rows,minlength=len(model.anchors.features))
    assert torch.equal(counts,model.anchors.member_counts) and ((counts>0)&(counts<=10)).all()
    metadata=load_selector_metadata(str(cloud/'selector_metadata.ply'),'cuda');assert (metadata['maturity']==1).all()
    with (out/(name+'_native.log')).open('w') as log:
        subprocess.run([str(repo/'bin/anchor_checkpoint_probe'),str(cp/'selector.pt'),str(cloud/'point_cloud.ply'),
            str(cloud/'selector_metadata.ply'),str(out/(name+'_native.pt')),'explicit_blocks'],stdout=log,stderr=subprocess.STDOUT,check=True)
    native=torch.jit.load(str(out/(name+'_native.pt')),map_location='cuda').state_dict()
    for i,r in enumerate([.01,.05,.1,.15]):
        mask,_=build_selector_mask(model,g,metadata,r,config,1.,False);ids=torch.nonzero(mask).flatten()
        assert torch.equal(mask.bool(),native['mask_'+str(i)])
        folder=root/'quality'/name/'cached_bernoulli'/str(int(r*100))
        assert np.array_equal(np.load(folder/'selected_rows.npz')['rows'],ids.cpu().numpy())
        with torch.no_grad():
            encoded=model.forward_selection_features(g.get_xyz,g.get_scaling,g.get_features_dc,g.get_opacity,torch.tensor([[r]],device='cuda'))
            residual=model.compensation(encoded[1][ids]);assert torch.isfinite(residual).all()
            attributes=compensated_attributes(model,g,mask,r,encoded[1])
        for attr,native_name in [('xyz','xyz'),('scaling','scale'),('rotation','rotation')]:
            assert torch.allclose(attributes[attr][ids],native['compensated_'+native_name+'_'+str(i)],atol=2e-6,rtol=2e-5)
        base_alpha=g.get_opacity[ids];alpha=base_alpha;delta_rgb=torch.zeros_like(g.get_features_dc[ids])
        if name=='appearance_v2':
            exported=GaussianModel(3);exported.load_ply(str(folder/'point_cloud.ply'))
            assert torch.allclose(attributes['features_dc'][ids],native['compensated_dc_'+str(i)],atol=2e-6,rtol=2e-5)
            assert torch.allclose(attributes['opacity_logits'][ids],native['compensated_opacity_logits_'+str(i)],atol=2e-6,rtol=2e-5)
            assert torch.equal(exported._opacity,attributes['opacity_logits'][ids])
            assert torch.equal(exported.get_features_dc,attributes['features_dc'][ids])
            assert torch.equal(exported.get_features_rest,g.get_features_rest[ids])
            prepared=prepare_render_data(g,mask,attributes)['inputs']
            assert torch.equal(prepared['opacities'],exported.get_opacity) and torch.equal(prepared['dc'],exported.get_features_dc)
            off=GaussianModel(3);off.load_ply(str(root/'quality/appearance_off/cached_bernoulli'/str(int(r*100))/'point_cloud.ply'))
            for field in ['_xyz','_scaling','_rotation','_features_rest']:
                assert torch.equal(getattr(off,field),getattr(exported,field)),field
            assert torch.equal(off._opacity,g._opacity[ids]) and torch.equal(off.get_features_dc,g.get_features_dc[ids])
            alpha=attributes['opacity_logits'][ids].sigmoid()
            delta_rgb=.28209479177387814*(attributes['features_dc'][ids]-g.get_features_dc[ids])
        statistics.append(dict(method=name,ratio=r,selected=len(ids),xyz_rms=float(residual[:,:3].square().mean().sqrt()),
            xyz_max=float(residual[:,:3].abs().max()),log_scale_rms=float(residual[:,3:6].square().mean().sqrt()),
            opacity_before_mean=float(base_alpha.mean()),opacity_after_mean=float(alpha.mean()),
            opacity_delta_mean=float((alpha-base_alpha).mean()),opacity_after_below_cutoff=int((alpha<1/255).sum()),
            delta_rgb_rms=float(delta_rgb.square().mean().sqrt()),delta_rgb_max=float(delta_rgb.abs().max())))
    row=dict(method=name,gaussians=len(g.get_xyz),anchors=len(counts),anchors_by_level=torch.bincount(model.anchors.levels,minlength=3).tolist(),
        full_renders=schedule['full_render_count'],lod_renders=schedule['lod_render_count'],
        gaussian_updates=sum(stages[x]['gaussian_updates'] for x in ['online','tail']),
        selector_updates=schedule['selector_online_lod_update_count']+schedule['selector_tail_lod_update_count'],
        online_end_mapping=schedule['online_end_mapping_iteration'],compensation_forwards=anchors['compensation_forwards'],
        cd_compensation_steps=anchors['cd_compensation_steps'],compensation_output_dim=width,
        selector_parameters=sum(p.numel() for key,p in model.named_parameters() if key!='anchors.features'),
        anchor_feature_values=model.anchors.features.numel())
    assert row['gaussian_updates']==row['full_renders']+row['lod_renders'];diagnostics.append(row)
    for file in ['ultrafast_rotation_state.json','mapping_stage_state.json']:(out/(name+'_'+file)).write_bytes((cp/file).read_bytes())
(out/'model_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
with (out/'compensation_statistics.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(statistics[0]));writer.writeheader();writer.writerows(statistics)
assert sha(jobs['geometry_v1']/'input_selector.yaml')==sha(jobs['appearance_v2']/'input_selector.yaml')
result=dict(passed=True,source_sha_verified=len(source),model_sha_unchanged=True,models=diagnostics,
    native_python_masks_exact=True,native_python_ply_appearance=True,renderer_matches_ply=True,
    appearance_off_same_geometry_and_selection=True,high_order_sh_exact=True,tf_disabled=True,input_yaml_exact=True,
    limitations='one scene/seed; online populations, poses and update counts differ; OFF is post-training ablation')
(out/'audit_summary.json').write_text(json.dumps(result,indent=2))
files=list(out.glob('*.*'))+list(root.glob('*.py'))+list(root.glob('*pipeline.log'))+[root/'source_manifest.json']
run=jobs['appearance_v2'];cp=run/'10000_shutdown/ply';cloud=cp/'point_cloud/iteration_10000'
files+=[run/'training.log',run/'input_selector.yaml',run/'CameraTrajectory_TUM.txt',cp/'selector.pt',cp/'selector_config.yaml',
    cloud/'point_cloud.ply',cloud/'selector_metadata.ply',cloud/'anchor_selector_state.json',cloud/'anchor_pruning_state.pt']
for folder in (root/'quality').glob('*/*/*'):
    files+=[p for p in folder.iterdir() if p.suffix in ['.txt','.json','.log','.npz','.ply']]
files=sorted(set(p for p in files if p.is_file()))
(root/'evidence_manifest.json').write_text(json.dumps({str(p.relative_to(root)):sha(p) for p in files},indent=2))
with tarfile.open(root/'evidence.tar.gz','w:gz') as archive:
    for p in files+[root/'evidence_manifest.json']:archive.add(p,arcname=str(p.relative_to(root)))
(root/'archive_info.json').write_text(json.dumps(dict(sha256=sha(root/'evidence.tar.gz'),bytes=(root/'evidence.tar.gz').stat().st_size)))
print('COLLECTED',len(files),flush=True)
