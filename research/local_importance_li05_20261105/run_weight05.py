"""只改变LI权重，完成短程验收和fr1_xyz完整Fine；保护已有结果。"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
PREVIOUS=Path('/home/prmi/zwj/experiments/local_importance_li_20261105')
STATE=ROOT/'reports/EXPERIMENT_STATUS.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(value):
    value['checked_unix']=time.time()
    temporary=STATE.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(STATE)


def identity(pid):
    return dict(pid=pid,start_ticks=Path(f'/proc/{pid}/stat').read_text().split()[21],
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def main():
    lock=(ROOT/'reports/experiment.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not STATE.exists(), '禁止覆盖已有实验状态'
    original=json.loads((PREVIOUS/'configs/full_li01.json').read_text())
    text=Path(original['selector_config']).read_text()
    assert text.count('ViewBudget.li_weight: 0.1')==1
    config=text.replace('ViewBudget.li_weight: 0.1','ViewBudget.li_weight: 0.5')
    selector=ROOT/'configs/li05.yaml';selector.write_text(config)
    assert config.replace('ViewBudget.li_weight: 0.5','ViewBudget.li_weight: 0.1')==text
    env=dict(os.environ)
    env['LD_LIBRARY_PATH']=':'.join([str(ROOT/'runtime/lib'),
        '/home/prmi/zwj/experiments/real_lod_full_joint_20261103/source/third_party/ORB-SLAM3/lib',
        '/home/prmi/zwj/mycartgs/third_party/install/opencv-cuda118-sm86-gcc9/lib',
        '/home/prmi/zwj/mycartgs/third_party/install/jsoncpp-1.9.5/lib',
        '/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/lib','/usr/local/cuda-11.8/lib64'])
    env['OMP_NUM_THREADS']='2'
    binary_files=['runtime/bin/lod_from_full','runtime/lib/libgaussian_mapper.so']
    binary={n:sha(ROOT/n) for n in binary_files}
    assert all(h==sha(PREVIOUS/n) for n,h in binary.items())
    (ROOT/'reports/BINARY_PARITY.json').write_text(json.dumps(dict(passed=True,sha256=binary),indent=2))
    state=dict(state='smoke',controller=identity(os.getpid()),li_weight=.5,steps=11580,
               binary_parity=True,smoke=None,groups={});save(state)

    def execute(name,manifest,steps=None):
        value=json.loads(manifest.read_text());output=Path(value['output'])
        assert not output.exists(), '保留已有模型及失败现场'
        for key,h in value['input_sha256'].items():assert sha(value[key])==h
        assert shutil.disk_usage(ROOT).free>5*1024**3
        command=[str(ROOT/'runtime/bin/lod_from_full'),str(manifest)]+([str(steps)] if steps else [])
        started=time.monotonic()
        with (ROOT/'logs'/f'{name}.log').open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            state['active']=dict(name=name,command=command,process=identity(child.pid),seconds=0,step=0);save(state)
            while child.poll() is None:
                state['active']['seconds']=time.monotonic()-started
                maps=Path(f'/proc/{child.pid}/maps')
                if maps.exists():
                    content=maps.read_text()
                    if 'libgaussian_mapper.so' in content:
                        assert str(ROOT/'runtime/lib/libgaussian_mapper.so') in content
                        (ROOT/'reports'/f'runtime_maps_{name}.txt').write_text(content)
                progress=output/'real_lod_steps.csv'
                if progress.exists():
                    with progress.open('rb') as stream:
                        stream.seek(max(0,progress.stat().st_size-2048));lines=stream.read().decode().splitlines()
                    state['active']['step']=next((int(s.split(',')[0]) for s in reversed(lines) if s.split(',')[0].isdigit()),0)
                save(state);time.sleep(3)
        assert child.returncode==0, f'{name}失败，检查日志，不自动重试'
        result=dict(returncode=0,seconds=time.monotonic()-started,output=str(output))
        fine=json.loads((output/'training_verified.json').read_text());assert fine['passed']
        return result

    def manifest(name):
        value=dict(original)
        for key in ['camera_config','mapping_config']:value[key]=str(ROOT/'configs'/Path(original[key]).name)
        value['selector_config']=str(selector);value['output']=str(ROOT/'runs'/name)
        value['input_sha256']={key:sha(value[key]) for key in original['input_sha256']}
        path=ROOT/'configs'/f'{name}.json';path.write_text(json.dumps(value,indent=2));return path

    state['smoke']=execute('smoke_li05',manifest('smoke_li05'),12);save(state)
    import torch
    def check_initial(folder,reference):
        a=torch.jit.load(str(folder/'step0_selector.pt'),map_location='cpu').state_dict()
        b=torch.jit.load(str(reference/'step0_selector.pt'),map_location='cpu').state_dict()
        assert set(a)==set(b) and len(a)==61
        for key in a:
            assert a[key].shape==b[key].shape and a[key].dtype==b[key].dtype
            assert torch.equal(a[key].contiguous().reshape(-1).view(torch.uint8),b[key].contiguous().reshape(-1).view(torch.uint8))
        assert sha(folder/'step0_master.ply')==sha(reference/'step0_master.ply')
        return dict(passed=True,tensors=len(a),bitwise=True,master_sha256=sha(folder/'step0_master.ply'))
    state['smoke_initial_parity']=check_initial(ROOT/'runs/smoke_li05',PREVIOUS/'runs/smoke_li01')
    assert (ROOT/'runs/smoke_li05/training_keyframes.csv').read_bytes()==(PREVIOUS/'runs/smoke_li01/training_keyframes.csv').read_bytes()
    state['state']='training';save(state)
    full_manifest=manifest('full_li05');result=execute('full_li05',full_manifest)
    folder=ROOT/'runs/full_li05'
    checkpoint=folder/'23560_shutdown/ply'
    fine=json.loads((checkpoint/'real_lod_state.json').read_text())
    for key in ['optimizer_commits','master_updates','selector_updates','full_training_renders',
                'lod_training_renders','view_budget_training_renders','view_budget_li_full_renders','view_budget_li_backwards']:
        assert fine[key]==11580,key
    assert fine['view_budget_li_weight']==.5 and fine['ratio_steps']==[1930]*6
    anchor=json.loads((checkpoint/'point_cloud/iteration_23560/anchor_selector_state.json').read_text())
    assert anchor['gaussian_count']==13298 and anchor['anchor_count']==2790
    assert [s['adam_step'] for s in anchor['compensation_feature_state']]==[11580,9650,7720,5790,3860,1930]
    state['full_initial_parity']=check_initial(folder,PREVIOUS/'runs/full_li01')
    assert (folder/'training_keyframes.csv').read_bytes()==(PREVIOUS/'runs/full_li01/training_keyframes.csv').read_bytes()
    shutil.copyfile(Path(original['full_ply']).parent.parent/'CameraTrajectory_TUM.txt',folder/'CameraTrajectory_TUM.txt')
    assert all(h==sha(ROOT/n) for n,h in binary.items())
    state.update(state='training_complete',active=None)
    state['groups']['li05']=dict(result,checkpoint=str(checkpoint),verified=fine);save(state)


if __name__=='__main__':
    try:main()
    except Exception as error:
        if STATE.exists():
            state=json.loads(STATE.read_text());state.update(state='failed',error=repr(error));save(state)
        raise
