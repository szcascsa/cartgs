"""复用已有baseline，仅训练两个视角组并审计初始化/训练曝光。"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
IMPLEMENTATION=ROOT
PREVIOUS=Path('/home/prmi/zwj/experiments/local_importance_li_20261105')
STATE=ROOT/'reports/TRAINING_STATUS.json'
GROUPS=['raw_no_li','raw_li01']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(pid):
    return dict(pid=pid,start_ticks=Path(f'/proc/{pid}/stat').read_text().split()[21],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def save(state):
    state['checked_unix']=time.time();p=STATE.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2));p.replace(STATE)


def main():
    lock=(ROOT/'reports/training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not STATE.exists(),'已有训练状态，禁止覆盖'
    assert json.loads((ROOT/'reports/INFERENCE_VALIDATION.json').read_text())['passed']
    binaries={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'runtime').rglob('*') if p.is_file()}
    assert all(h==sha(IMPLEMENTATION/name) for name,h in binaries.items())
    (ROOT/'reports/BINARY_PARITY.json').write_text(json.dumps(dict(passed=True,sha256=binaries),indent=2))
    env=dict(os.environ);env['OMP_NUM_THREADS']='2'
    env['LD_LIBRARY_PATH']=':'.join([str(ROOT/'runtime/lib'),
        '/home/prmi/zwj/experiments/real_lod_full_joint_20261103/source/third_party/ORB-SLAM3/lib',
        '/home/prmi/zwj/mycartgs/third_party/install/opencv-cuda118-sm86-gcc9/lib',
        '/home/prmi/zwj/mycartgs/third_party/install/jsoncpp-1.9.5/lib',
        '/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/lib','/usr/local/cuda-11.8/lib64'])
    previous=json.loads((PREVIOUS/'reports/EXPERIMENT_STATUS.json').read_text())['groups']['no_li']
    reused=dict(previous,reused=True,source_root=str(PREVIOUS))
    state=dict(state='training',controller=identity(os.getpid()),groups={'baseline':reused},target_steps=11580);save(state)
    for group in GROUPS:
        template=group
        value=json.loads((IMPLEMENTATION/'configs'/f'{template}.json').read_text())
        for key in ['selector_config','mapping_config','camera_config']:
            value[key]=str(ROOT/'configs'/Path(value[key]).name)
        value['output']=str(ROOT/'runs'/f'full_{group}');output=Path(value['output'])
        assert not output.exists()
        for key,h in value['input_sha256'].items():assert sha(Path(value[key]))==h,key
        manifest=ROOT/'configs'/f'full_{group}.json';manifest.write_text(json.dumps(value,indent=2))
        assert shutil.disk_usage(ROOT).free>5*1024**3
        command=[str(ROOT/'runtime/bin/lod_from_full'),str(manifest)]
        started=time.monotonic()
        with (ROOT/'logs'/f'full_{group}.log').open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            state['active']=dict(group=group,process=identity(child.pid),step=0,seconds=0);save(state)
            while child.poll() is None:
                maps=Path(f'/proc/{child.pid}/maps')
                if maps.exists():
                    text=maps.read_text()
                    if 'libgaussian_mapper.so' in text:
                        assert str(ROOT/'runtime/lib/libgaussian_mapper.so') in text
                        (ROOT/'reports'/f'runtime_maps_{group}.txt').write_text(text)
                state['active']['seconds']=time.monotonic()-started
                progress=output/'real_lod_steps.csv'
                if progress.exists():
                    with progress.open('rb') as stream:
                        stream.seek(max(0,progress.stat().st_size-2048));lines=stream.read().decode().splitlines()
                    state['active']['step']=next((int(s.split(',')[0]) for s in reversed(lines) if s.split(',')[0].isdigit()),0)
                save(state);time.sleep(3)
        assert child.returncode==0,f'{group}失败，检查日志；不自动重试'
        assert json.loads((output/'training_verified.json').read_text())['passed']
        checkpoint=output/'23560_shutdown/ply';fine=json.loads((checkpoint/'real_lod_state.json').read_text())
        for key in ['optimizer_commits','master_updates','selector_updates','full_training_renders',
                    'lod_training_renders','view_budget_training_renders']:
            assert fine[key]==11580,key
        assert fine['ratio_steps']==fine['view_budget_ratio_steps']==[1930]*6
        assert fine['view_budget_view_residual']==(group!='baseline')
        assert fine['view_budget_li_weight']==(float(.10000000149011612) if group=='raw_li01' else 0)
        assert fine['view_budget_li_full_renders']==fine['view_budget_li_backwards']==(11580 if group=='raw_li01' else 0)
        assert fine['view_budget_residual_input']=='physical11' and fine['view_budget_residual_parameters']==321
        assert fine['lod_nonfinite']==0
        anchor=json.loads((checkpoint/'point_cloud/iteration_23560/anchor_selector_state.json').read_text())
        assert anchor['gaussian_count']==13298 and anchor['anchor_count']==2790
        assert [s['adam_step'] for s in anchor['compensation_feature_state']]==[11580,9650,7720,5790,3860,1930]
        shutil.copyfile(Path(value['full_ply']).parent.parent/'CameraTrajectory_TUM.txt',output/'CameraTrajectory_TUM.txt')
        state['groups'][group]=dict(seconds=time.monotonic()-started,checkpoint=str(checkpoint),verified=fine);save(state)
    import torch
    reference=None;master=None;sequence=None;branch=None
    for group in ['baseline']+GROUPS:
        folder=PREVIOUS/'runs/full_no_li' if group=='baseline' else ROOT/'runs'/f'full_{group}'
        values=torch.jit.load(str(folder/'step0_selector.pt'),map_location='cpu').state_dict()
        current=[(r['step'],r['keyframe_id'],r['ratio']) for r in map(json.loads,(folder/'view_budget_steps.jsonl').read_text().splitlines())]
        if reference is None:reference=values;master=sha(folder/'step0_master.ply');sequence=current
        assert len(values)==(61 if group=='baseline' else 68)
        for key,a in reference.items():
            b=values[key];assert a.shape==b.shape and a.dtype==b.dtype
            assert torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8)),key
        assert current==sequence and master==sha(folder/'step0_master.ply')
        if group!='baseline':
            extra={k:v for k,v in values.items() if k not in reference}
            if branch is None:branch=extra
            assert all(torch.equal(branch[k],v) for k,v in extra.items())
    assert all(sha(ROOT/name)==h for name,h in binaries.items())
    (ROOT/'reports/INITIALIZATION_PARITY.json').write_text(json.dumps(dict(passed=True,original_tensors_bitwise=61,
        new_branch_initial_states_bitwise=True,master_sha256=master,identical_camera_ratio_steps=11580),indent=2))
    state.update(state='training_complete',active=None);save(state)


if __name__=='__main__':
    try:main()
    except Exception as error:
        if STATE.exists():
            state=json.loads(STATE.read_text());state.update(state='failed',error=repr(error));save(state)
        raise
