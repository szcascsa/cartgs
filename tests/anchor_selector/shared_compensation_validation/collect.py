"""核对最终 checkpoint、真实导出属性和训练时钟，保存可复算证据。"""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import numpy as np
import torch

root=Path('/home/prmi/zwj/shared-selector-compensation-20261021');repo=root/'source';out=root/'validation'
sys.path.insert(0,str(repo/'python'))
from selector import load_selector,load_selector_metadata,build_selector_mask,compensated_attributes
from transform_field import load_transform_field,transformed_attributes
from gaussian_model import GaussianModel
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=json.loads((out/'evaluation_plan.json').read_text())
assert json.loads((out/'quality_complete.json').read_text())['completed']
assert all(sha(p)==h for p,h in plan['model_sha256'].items())
manifest=json.loads((root/'source_manifest.json').read_text())
assert all(sha(repo/p)==h for p,h in manifest.items())
jobs=dict(tf_baseline=Path('/home/prmi/zwj/variable-anchor-joint-quality-20261021/variable_seed2026'),
          shared_compensation=root/'candidate_seed2026')
diagnostics=[];residuals=[]
config=dict(drift_temperature=.1,beta_fast=.6,beta_slow=.15,compatibility_mature_threshold=.8,entry_maturity=0)
for name,run in jobs.items():
    cp=run/'10000_shutdown/ply';cloud=cp/'point_cloud/iteration_10000'
    schedule=json.loads((cp/'ultrafast_rotation_state.json').read_text())
    stages=json.loads((cp/'mapping_stage_state.json').read_text())
    anchors=json.loads((cloud/'anchor_selector_state.json').read_text())
    assert not schedule['serial_full_lod'] and schedule['post_topology_full_steps']==0 and schedule['balanced']
    assert schedule['ultrafast_total_normal_steps']==10000
    assert sum(stages[x]['lod_nonfinite'] for x in ['online','tail'])==0
    model=load_selector(str(cp/'selector.pt'),'cuda');g=GaussianModel(3);g.load_ply(str(cloud/'point_cloud.ply'))
    metadata=load_selector_metadata(str(cloud/'selector_metadata.ply'),'cuda')
    counts=torch.bincount(model.anchors.rows,minlength=len(model.anchors.features))
    assert torch.equal(counts,model.anchors.member_counts) and ((counts>0)&(counts<=10)).all()
    assert (metadata['maturity']==1).all()
    with (out/(name+'_native.log')).open('w') as log:
        subprocess.run([str(repo/'bin/anchor_checkpoint_probe'),str(cp/'selector.pt'),str(cloud/'point_cloud.ply'),
            str(cloud/'selector_metadata.ply'),str(out/(name+'_native.pt')),'explicit_blocks'],stdout=log,stderr=subprocess.STDOUT,check=True)
    native=torch.jit.load(str(out/(name+'_native.pt')),map_location='cuda').state_dict()
    for i,r in enumerate([.01,.05,.1,.15]):
        mask,_=build_selector_mask(model,g,metadata,r,config,1.,False)
        assert torch.equal(mask.bool(),native['mask_'+str(i)])
        ids=torch.nonzero(mask).flatten()
        saved=np.load(root/'quality'/name/'cached_bernoulli'/str(int(r*100))/'selected_rows.npz')['rows']
        assert np.array_equal(saved,ids.cpu().numpy())
        if name=='shared_compensation':
            with torch.no_grad():
                encoded=model.forward_selection_features(g.get_xyz,g.get_scaling,g.get_features[:,:1],g.get_opacity,torch.tensor([[r]],device='cuda'))
                residual=model.compensation(encoded[1][ids])
                assert torch.isfinite(residual).all()
                attributes=compensated_attributes(model,g,mask,r,encoded[1])
            exported=GaussianModel(3)
            exported.load_ply(str(root/'quality'/name/'cached_bernoulli'/str(int(r*100))/'point_cloud.ply'))
            for attr,actual in [('xyz',exported.get_xyz),('scaling',exported.get_scaling),('rotation',exported.get_rotation)]:
                assert torch.allclose(attributes[attr][ids],native['compensated_'+{'scaling':'scale'}.get(attr,attr)+'_'+str(i)],atol=2e-6,rtol=2e-5)
                assert torch.allclose(actual,attributes[attr][ids],atol=2e-6,rtol=2e-5)
            assert torch.equal(exported.get_opacity,g.get_opacity[ids])
            residuals.append(dict(ratio=r,selected=len(ids),xyz_rms=float(residual[:,:3].square().mean().sqrt()),
                xyz_max=float(residual[:,:3].abs().max()),log_scale_rms=float(residual[:,3:6].square().mean().sqrt()),
                rotation_raw_rms=float(residual[:,6:].square().mean().sqrt())))
    if name=='shared_compensation':
        assert not (cp/'transform_field.pt').exists() and not (cp/'transform_state.json').exists()
        assert schedule['tf_online_lod_update_count']==schedule['tf_tail_lod_update_count']==0
        assert anchors['compensation_forwards']>0 and model.compensation[-1].weight.abs().sum()>0
    tf=None if name=='shared_compensation' else load_transform_field(str(cp),'cuda')
    # 包含选择、gather、选后补偿；各自真实 N/K，不能视作固定人口的纯网络加速比。
    @torch.no_grad()
    def inference():
        if name=='shared_compensation':
            encoded=model.forward_selection_features(g.get_xyz,g.get_scaling,g.get_features[:,:1],g.get_opacity,torch.tensor([[.1]],device='cuda'))
            mask,_=build_selector_mask(model,g,metadata,.1,config,1.,False,forward_features=encoded)
            return compensated_attributes(model,g,mask,.1,encoded[1])
        mask,_=build_selector_mask(model,g,metadata,.1,config,1.,False)
        return transformed_attributes(tf,g,mask,metadata['maturity'],.1)
    for _ in range(10):inference()
    times=[];torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();memory=torch.cuda.memory_allocated()
    for repeat in range(5):
        start=time.perf_counter()
        for _ in range(50):inference()
        torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000/50)
    row=dict(method=name,gaussians=len(g.get_xyz),anchors=len(counts),anchors_by_level=torch.bincount(model.anchors.levels,minlength=3).tolist(),
        full_renders=schedule['full_render_count'],lod_renders=schedule['lod_render_count'],
        gaussian_updates=sum(stages[x]['gaussian_updates'] for x in ['online','tail']),
        selector_updates=schedule['selector_online_lod_update_count']+schedule['selector_tail_lod_update_count'],
        tf_updates=schedule['tf_online_lod_update_count']+schedule['tf_tail_lod_update_count'],
        online_end_mapping=schedule['online_end_mapping_iteration'],compensation_forwards=anchors.get('compensation_forwards',0),
        cd_compensation_steps=anchors.get('cd_compensation_steps',0),
        selector_parameters=sum(p.numel() for key,p in model.named_parameters() if key!='anchors.features'),
        anchor_feature_values=model.anchors.features.numel(),tf_parameters=0 if tf is None else sum(p.numel() for p in tf.parameters()),
        inference_ms_mean=float(np.mean(times)),inference_ms_repeats=times,inference_peak_extra_bytes=torch.cuda.max_memory_allocated()-memory)
    assert row['gaussian_updates']==row['full_renders']+row['lod_renders']
    diagnostics.append(row)
    for file in ['ultrafast_rotation_state.json','mapping_stage_state.json']:(out/(name+'_'+file)).write_bytes((cp/file).read_bytes())
(out/'model_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
with (out/'compensation_statistics.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(residuals[0]));writer.writeheader();writer.writerows(residuals)
result=dict(passed=True,models=diagnostics,source_sha_verified=len(manifest),model_sha_unchanged=True,
    native_python_masks_exact=True,native_python_export_compensated_attributes=True,tf_disabled=True,
    same_selected_compensation_diagnostic=True,scene='fr1_xyz',seed=2026,
    limitations='one scene and seed; different online map population, poses and update counts; HexPlane regularization removed with TF')
(out/'audit_summary.json').write_text(json.dumps(result,indent=2))
files=list(out.glob('*.*'))+list(root.glob('*.py'))+list(root.glob('*pipeline.log'))+[root/'source_manifest.json']
run=jobs['shared_compensation'];cp=run/'10000_shutdown/ply';cloud=cp/'point_cloud/iteration_10000'
files+=[run/'training.log',run/'input_selector.yaml',run/'CameraTrajectory_TUM.txt',cp/'selector.pt',cp/'selector_config.yaml',
        cloud/'point_cloud.ply',cloud/'selector_metadata.ply',cloud/'anchor_pruning_state.pt',cloud/'anchor_selector_state.json']
for folder in (root/'quality').glob('*/*/*'):
    files+=[p for p in folder.iterdir() if p.suffix in ['.txt','.json','.log','.npz','.ply']]
files=sorted(set(p for p in files if p.is_file()))
manifest={str(p.relative_to(root)):sha(p) for p in files}
(root/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(root/'evidence.tar.gz','w:gz') as archive:
    for p in files+[root/'evidence_manifest.json']:archive.add(p,arcname=str(p.relative_to(root)))
(root/'archive_info.json').write_text(json.dumps(dict(sha256=sha(root/'evidence.tar.gz'),bytes=(root/'evidence.tar.gz').stat().st_size),indent=2))
print('COLLECTED',len(files),flush=True)
