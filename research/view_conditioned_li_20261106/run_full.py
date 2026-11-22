"""仅训练三组LI=0.1，匹配无LI结果只读复用。"""
import json
import os
from pathlib import Path
import shutil
from common import ROOT,BASE,GROUPS,SCHEMAS,sha,save,identity,run


def main():
    path=ROOT/'reports/TRAINING_STATUS.json';assert not path.exists()
    assert json.loads((ROOT/'reports/INFERENCE_VALIDATION.json').read_text())['passed']
    state=dict(state='training',controller=identity(os.getpid()),groups={},target_steps=11580);save(path,state)
    binaries=json.loads((ROOT/'reports/BUILD_HASHES.json').read_text());assert all(sha(ROOT/n)==h for n,h in binaries.items())
    for group in GROUPS:
        value=json.loads((ROOT/'configs'/f'{group}.json').read_text());value['output']=str(ROOT/'runs'/f'full_{group}');output=Path(value['output'])
        assert not output.exists() and shutil.disk_usage(ROOT).free>5*1024**3
        assert all(sha(value[k])==h for k,h in value['input_sha256'].items())
        manifest=ROOT/'configs'/f'full_{group}.json';save(manifest,value)
        timing=run('full_'+group,[str(ROOT/'runtime/bin/lod_from_full'),str(manifest)],state,path,output)
        assert json.loads((output/'training_verified.json').read_text())['passed']
        checkpoint=output/'23560_shutdown/ply';fine=json.loads((checkpoint/'real_lod_state.json').read_text())
        for key in ['optimizer_commits','master_updates','selector_updates','full_training_renders','lod_training_renders','view_budget_training_renders']:assert fine[key]==11580,key
        assert fine['ratio_steps']==fine['view_budget_ratio_steps']==[1930]*6
        assert fine['view_budget_conditioning']==SCHEMAS[group] and abs(fine['view_budget_li_weight']-.1)<1e-6 and fine['view_budget_li_backwards']==11580 and fine['view_budget_li_full_renders']==11580 and fine['lod_nonfinite']==0
        anchor=json.loads((checkpoint/'point_cloud/iteration_23560/anchor_selector_state.json').read_text())
        assert anchor['gaussian_count']==13298 and anchor['anchor_count']==2790
        assert [s['adam_step'] for s in anchor['compensation_feature_state']]==[11580,9650,7720,5790,3860,1930]
        shutil.copyfile(Path(value['full_ply']).parent.parent/'CameraTrajectory_TUM.txt',output/'CameraTrajectory_TUM.txt')
        state['groups'][group]=dict(**timing,checkpoint=str(checkpoint),verified=fine);save(path,state)
    import torch
    for group in GROUPS:
        values=torch.jit.load(str(ROOT/'runs'/f'full_{group}'/'23560_shutdown/ply/selector.pt'),map_location='cpu').state_dict()
        selector=values['local_emd.0.weight'][:,11:18]
        compensation=values['spatial_compensation.view_projection.weight'][:,:7]
        assert bool(selector.count_nonzero())==(group in ['selector_li','both_li'])
        assert bool(compensation.count_nonzero())==(group in ['compensator_li','both_li'])
    baseline=BASE/'runs/full_no_li'
    reference=torch.jit.load(str(baseline/'step0_selector.pt'),map_location='cpu').state_dict()
    sequence=[(v['step'],v['keyframe_id'],v['ratio']) for v in map(json.loads,(baseline/'view_budget_steps.jsonl').read_text().splitlines())]
    for group in GROUPS:
        folder=ROOT/'runs'/f'full_{group}';values=torch.jit.load(str(folder/'step0_selector.pt'),map_location='cpu').state_dict()
        for n,a in reference.items():
            b=values[n][:,:11] if n=='local_emd.0.weight' else values[n]
            assert torch.equal(a.reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8)),n
        assert sha(folder/'step0_master.ply')==sha(baseline/'step0_master.ply')
        actual=[(v['step'],v['keyframe_id'],v['ratio']) for v in map(json.loads,(folder/'view_budget_steps.jsonl').read_text().splitlines())]
        assert actual==sequence and len(actual)==11580
    assert all(sha(ROOT/n)==h for n,h in binaries.items())
    save(ROOT/'reports/INITIALIZATION_PARITY.json',dict(passed=True,original_tensor_values_bitwise=61,expanded_first_layer_original_columns_bitwise=True,
        master_sha256=sha(baseline/'step0_master.ply'),identical_camera_ratio_steps=11580))
    state.update(state='training_complete',active=None);save(path,state)


if __name__=='__main__':
    try:main()
    except Exception as error:
        p=ROOT/'reports/TRAINING_STATUS.json';v=json.loads(p.read_text());v.update(state='failed',error=repr(error));save(p,v);raise
