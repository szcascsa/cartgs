"""独立构建完成后串行验收；不启动完整训练，不覆盖旧输出。"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
PREVIOUS=Path('/home/prmi/zwj/experiments/local_importance_li_20261105')
STATE=ROOT/'reports/VALIDATION_STATUS.json'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(state):
    state['checked_unix']=time.time();p=STATE.with_suffix('.tmp')
    p.write_text(json.dumps(state,indent=2));p.replace(STATE)


def identity(pid):
    return dict(pid=pid,start_ticks=Path(f'/proc/{pid}/stat').read_text().split()[21],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def main():
    lock=(ROOT/'reports/validation.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not STATE.exists(),'禁止覆盖已有验收'
    state=dict(state='waiting_build',controller=identity(os.getpid()),tests=[],smoke={});save(state)
    build=json.loads((ROOT/'reports/BUILD_PID.json').read_text())
    while not (ROOT/'reports/BUILD_HASHES.json').exists():
        p=Path(f'/proc/{build["pid"]}/stat')
        assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==build['boot_id']
        assert p.exists() and p.read_text().split()[21]==build['start_ticks'],'构建退出且未成功，检查日志'
        time.sleep(3)
    env=dict(os.environ);env['OMP_NUM_THREADS']='2'
    env['LD_LIBRARY_PATH']=':'.join([str(ROOT/'runtime/lib'),
        '/home/prmi/zwj/experiments/real_lod_full_joint_20261103/source/third_party/ORB-SLAM3/lib',
        '/home/prmi/zwj/mycartgs/third_party/install/opencv-cuda118-sm86-gcc9/lib',
        '/home/prmi/zwj/mycartgs/third_party/install/jsoncpp-1.9.5/lib',
        '/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/lib','/usr/local/cuda-11.8/lib64'])

    def run(name,command):
        started=time.monotonic()
        with (ROOT/'logs'/f'{name}.log').open('w') as log:
            p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            state['active']=dict(name=name,process=identity(p.pid));save(state)
            while p.poll() is None:
                maps=Path(f'/proc/{p.pid}/maps')
                if maps.exists():
                    text=maps.read_text()
                    if 'libgaussian_mapper.so' in text:
                        assert str(ROOT/'runtime/lib/libgaussian_mapper.so') in text
                        (ROOT/'reports'/f'runtime_maps_{name}.txt').write_text(text)
                time.sleep(1)
        assert p.returncode==0,f'{name}失败；保留现场，不自动重试'
        return dict(name=name,returncode=0,seconds=time.monotonic()-started)

    state['state']='native_tests';save(state)
    tests=[('residual_'+m,'view_budget_residual_test',[m,str(ROOT/'runs'/('test_residual_'+m))])
        for m in ['unit','isolation','on','no_li','empty']]
    tests += [('legacy_li_'+m,'local_importance_test',[m,str(ROOT/'runs'/('test_legacy_li_'+m))])
        for m in ['unit','isolation','on','off']]
    tests += [('legacy_budget_'+m,'view_budget_training_test',[m,str(ROOT/'runs'/('test_legacy_budget_'+m))])
        for m in ['sampling','joint','empty','health']]
    tests += [('legacy_detached_'+m,'view_residual_test',[m,str(ROOT/'runs'/('test_legacy_detached_'+m+'.pt'))])
        for m in ['cpu','cuda']]
    tests += [('global_lod','real_lod_test',[str(ROOT/'runs/test_global_lod')])]
    for name,binary,args in tests:
        state['tests'].append(run(name,[str(ROOT/'runtime/bin'/binary)]+args));save(state)
    state['state']='checkpoint_smoke';save(state)
    original=json.loads((PREVIOUS/'configs/full_li01.json').read_text())
    config=(PREVIOUS/'configs/li01.yaml').read_text()
    for group,residual,weight in [('off',False,0),('view_no_li',True,0),('view_li01',True,.1)]:
        value=dict(original)
        selector=ROOT/'configs'/f'{group}.yaml'
        text=config.replace('ViewBudget.li_weight: 0.1',f'ViewBudget.li_weight: {weight}')
        selector.write_text(text+f'\nViewBudget.view_residual: {int(residual)}\n')
        value['selector_config']=str(selector);value['output']=str(ROOT/'runs'/f'smoke_{group}')
        assert not Path(value['output']).exists()
        for key in ['camera_config','mapping_config']:value[key]=str(ROOT/'configs'/Path(original[key]).name)
        value['input_sha256']={key:sha(Path(value[key])) for key in original['input_sha256']}
        manifest=ROOT/'configs'/f'{group}.json';manifest.write_text(json.dumps(value,indent=2))
        state['smoke'][group]=run('smoke_'+group,[str(ROOT/'runtime/bin/lod_from_full'),str(manifest),'36']);save(state)
    import torch
    base=ROOT/'runs/smoke_off'
    a=torch.jit.load(str(base/'step0_selector.pt'),map_location='cpu').state_dict();assert len(a)==61
    first_reference=json.loads((base/'view_budget_steps.jsonl').read_text().splitlines()[0])
    for group in ['off','view_no_li','view_li01']:
        folder=ROOT/'runs'/f'smoke_{group}'
        b=torch.jit.load(str(folder/'step0_selector.pt'),map_location='cpu').state_dict()
        for key in a:
            assert a[key].shape==b[key].shape and a[key].dtype==b[key].dtype,key
            assert torch.equal(a[key].contiguous().reshape(-1).view(torch.uint8),b[key].contiguous().reshape(-1).view(torch.uint8)),key
        assert len(b)==(61 if group=='off' else 68)
        assert sha(folder/'step0_master.ply')==sha(base/'step0_master.ply')
        assert (folder/'training_keyframes.csv').read_bytes()==(base/'training_keyframes.csv').read_bytes()
        assert json.loads((folder/'training_verified.json').read_text())['passed']
        fine=json.loads((folder/'12016_shutdown/ply/real_lod_state.json').read_text())
        for key in ['full_training_renders','lod_training_renders','view_budget_training_renders','optimizer_commits','master_updates','selector_updates']:
            assert fine[key]==36,key
        assert fine['view_budget_view_residual']==(group!='off')
        assert fine['ratio_steps']==fine['view_budget_ratio_steps']==[6]*6
        anchor=json.loads((folder/'12016_shutdown/ply/point_cloud/iteration_12016/anchor_selector_state.json').read_text())
        assert anchor['gaussian_count']==13298 and anchor['anchor_count']==2790
        assert [s['adam_step'] for s in anchor['compensation_feature_state']]==[36,30,24,18,12,6]
        first=json.loads((folder/'view_budget_steps.jsonl').read_text().splitlines()[0])
        for key in ['keyframe_id','ratio','N','candidate_N','eligible_N','K','global_K','photo_loss','budget_loss']:
            assert first[key]==first_reference[key],key
    state.update(state='native_smoke_complete',active=None,initial_original_tensors_bitwise=61,
        added_state_tensors=7,camera_sequence_identical=True,smoke_steps=36);save(state)


if __name__=='__main__':
    try:main()
    except Exception as error:
        if STATE.exists():
            state=json.loads(STATE.read_text());state.update(state='failed',error=repr(error));save(state)
        raise
