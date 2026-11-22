"""等待独立构建，验证无Other与旧路径，再从原Full训练一组。"""
import json
import math
import os
from pathlib import Path
import shutil
import time
from common import ROOT,HERE,PYTHON,run,sha,save,identity,environment

CONTROL=Path('/home/prmi/zwj/experiments/view_conditioned_li_20261106')


def initialization(folder,steps):
    import torch
    ref=CONTROL/'runs/full_both_li'
    a=torch.jit.load(str(ref/'step0_selector.pt'),map_location='cpu').state_dict()
    b=torch.jit.load(str(folder/'step0_selector.pt'),map_location='cpu').state_dict()
    removed=[n for n in a if 'spatial_compensation.heads.other.' in n]
    assert len(removed)==4 and len(b)==len(a)-3
    for n,v in a.items():
        if n in removed:continue
        assert torch.equal(v.contiguous().reshape(-1).view(torch.uint8),b[n].contiguous().reshape(-1).view(torch.uint8)),n
    assert int(b['spatial_compensation.geometry_head_schema'])==1
    assert sha(folder/'step0_master.ply')==sha(ref/'step0_master.ply')
    actual=[(v['step'],v['keyframe_id'],v['ratio']) for v in map(json.loads,(folder/'view_budget_steps.jsonl').read_text().splitlines())]
    expected=[(v['step'],v['keyframe_id'],v['ratio']) for v in map(json.loads,(ref/'view_budget_steps.jsonl').read_text().splitlines())][:steps]
    assert actual==expected and len(actual)==steps
    first=json.loads((folder/'view_budget_steps.jsonl').read_text().splitlines()[0]);old=json.loads((ref/'view_budget_steps.jsonl').read_text().splitlines()[0])
    fields=['photo_loss','budget_loss','global_loss','full_loss','li_loss','joint_loss','K','global_K']
    assert all(first[n]==old[n] for n in fields),{n:[first[n],old[n]] for n in fields}
    return dict(passed=True,common_state_tensors=len(a)-4,removed_parameter_tensors=removed,initial_master_sha256=sha(folder/'step0_master.ply'),
        identical_camera_ratio_steps=steps,first_step_losses_bitwise=True)


def verify(folder,steps):
    f=folder/f'{11980+steps}_shutdown/ply';v=json.loads((f/'real_lod_state.json').read_text())
    assert json.loads((folder/'training_verified.json').read_text())['passed']
    for k in ['optimizer_commits','master_updates','selector_updates','full_training_renders','lod_training_renders','view_budget_training_renders',
              'view_budget_li_backwards','view_budget_li_full_renders']:assert v[k]==steps,k
    assert v['view_budget_conditioning']==2 and abs(v['view_budget_li_weight']-.1)<1e-6 and v['lod_nonfinite']==0
    assert v['ratio_steps']==v['view_budget_ratio_steps']==[steps//6]*6
    anchor=json.loads((f/f'point_cloud/iteration_{11980+steps}/anchor_selector_state.json').read_text())
    assert anchor['gaussian_count']==13298 and anchor['anchor_count']==2790 and anchor['geometry_head_variant']==1
    assert [x['adam_step'] for x in anchor['compensation_feature_state']]==[steps//6*k for k in range(6,0,-1)]
    return v


def main():
    path=ROOT/'reports/TRAINING_STATUS.json';assert not path.exists()
    state=dict(state='waiting_build',controller=identity(os.getpid()),tests=[]);save(path,state)
    while not (ROOT/'reports/BUILD_HASHES.json').exists():time.sleep(3)
    state['state']='validating';save(path,state)
    tests=[('no_other_unit','residual48_no_other_test',[str(ROOT/'runs/native_unit')]),
           ('old_joint_unit','joint_view_conditioning_test',['unit',str(ROOT/'runs/old_unit')]),
           ('no_other_mapping','joint_view_conditioning_test',['view_li_no_other',str(ROOT/'runs/native_mapping')]),
           ('li_isolation','joint_view_conditioning_test',['view_li_isolation',str(ROOT/'runs/li_isolation')]),
           ('old_li_mapping','joint_view_conditioning_test',['view_li',str(ROOT/'runs/old_li_mapping')])]
    for name,binary,args in tests:state['tests'].append(dict(name=name,**run(name,[str(ROOT/'runtime/bin'/binary)]+args,state,path)));save(path,state)
    import torch
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    import sys
    sys.path[:0]=[str(ROOT/'source/python')]
    from selector import load_selector
    fixture_checks=[]
    for device in ['cpu','cuda']:
        directory=ROOT/'runs/native_unit'/device
        model=load_selector(str(directory/'selector.pt'),device);f=torch.jit.load(str(directory/'fixture.pt'),map_location=device).state_dict()
        for local in [False,True]:
            kwargs=dict(log_scale_input=True,anchor_feature=f['feature'])
            if local:kwargs['view_condition']=f['condition']
            out=model.spatial_compensation(*(f[n] for n in ['xyz','scale','rotation','dc','opacity','center','reference']),.05,**kwargs)
            expected=f['local_output' if local else 'global_output']
            assert torch.equal(out.detach().contiguous().view(torch.uint8),expected.contiguous().view(torch.uint8))
            fixture_checks.append(dict(device=device,local=local,bitwise=True))
    save(ROOT/'reports/PYTHON_PARITY.json',dict(passed=True,cases=fixture_checks))
    template=json.loads((CONTROL/'configs/full_both_li.json').read_text())
    for k in ['camera_config','mapping_config']:
        target=ROOT/'configs'/Path(template[k]).name;shutil.copyfile(template[k],target);template[k]=str(target)
    template['selector_config']=str(ROOT/'source/cfg/selector/spatial_offset3_residual48_joint_view_li_no_other.yaml')
    template['source_commit']='26c355b + no-Other patch';template['source_snapshot']='reports/SOURCE_FINAL_HASHES.json'
    template['input_sha256']={k:sha(template[k]) for k in template['input_sha256']}
    protected={}
    for checkpoint in [Path(template['full_ply']),CONTROL/'runs/full_both_li/23560_shutdown/ply']:
        for p in checkpoint.rglob('*'):
            if p.is_file():protected[str(p)]=sha(p)
    for p in [CONTROL/'evaluation/reports/PER_VIEW.csv',CONTROL/'evaluation/reports/SUMMARY.json',CONTROL/'runs/full_both_li/view_budget_steps.jsonl']:
        protected[str(p)]=sha(p)
    save(ROOT/'reports/HISTORICAL_INPUTS.json',protected)
    for name,steps in [('smoke_no_other',36),('full_no_other',11580)]:
        folder=ROOT/'runs'/name;assert not folder.exists()
        value=dict(template,output=str(folder));manifest=ROOT/'configs'/f'{name}.json';save(manifest,value)
        state.update(state=name,target_steps=steps);save(path,state)
        timing=run(name,[str(ROOT/'runtime/bin/lod_from_full'),str(manifest)]+(['36'] if steps==36 else []),state,path,folder)
        verified=verify(folder,steps);init=initialization(folder,steps)
        shutil.copyfile(Path(template['full_ply']).parent.parent/'CameraTrajectory_TUM.txt',folder/'CameraTrajectory_TUM.txt')
        state[name]=dict(**timing,verified=verified,initialization=init);save(path,state)
    assert all(sha(n)==h for n,h in protected.items())
    state.update(state='training_complete',active=None);save(path,state)
    save(ROOT/'reports/INITIALIZATION_PARITY.json',state['full_no_other']['initialization'])
    print('TRAINING_COMPLETE',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        p=ROOT/'reports/TRAINING_STATUS.json';state=json.loads(p.read_text());state.update(state='failed',error=repr(error));save(p,state);raise
