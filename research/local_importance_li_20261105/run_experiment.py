"""先验证再串行运行LI消融；失败立即保留现场，不覆盖历史实验。"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
OLD = Path('/home/prmi/zwj/experiments/global_view_joint_lod_20261105')
STATE = ROOT/'reports/EXPERIMENT_STATUS.json'


def save(value):
    value['checked_unix'] = time.time()
    temporary = STATE.with_suffix('.tmp'); temporary.write_text(json.dumps(value, indent=2)); temporary.replace(STATE)


def identity(pid):
    return {'pid':pid, 'start_ticks':Path(f'/proc/{pid}/stat').read_text().split()[21],
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    lock = (ROOT/'reports/experiment.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not STATE.exists(), '已有实验状态，禁止覆盖'
    assert (ROOT/'reports/BUILD_HASHES.json').exists()
    state = {'state':'validation', 'controller':identity(os.getpid()), 'tests':[], 'smoke':{}, 'groups':{}}
    save(state)
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = ':'.join([str(ROOT/'runtime/lib'),
        '/home/prmi/zwj/experiments/real_lod_full_joint_20261103/source/third_party/ORB-SLAM3/lib',
        '/home/prmi/zwj/mycartgs/third_party/install/opencv-cuda118-sm86-gcc9/lib',
        '/home/prmi/zwj/mycartgs/third_party/install/jsoncpp-1.9.5/lib',
        '/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/lib','/usr/local/cuda-11.8/lib64'])
    env['OMP_NUM_THREADS'] = '2'

    def execute(name, command, output=None):
        started = time.monotonic()
        with (ROOT/'logs'/f'{name}.log').open('w') as log:
            child = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            state['active'] = dict(name=name, command=command, process=identity(child.pid), seconds=0)
            save(state)
            while child.poll() is None:
                state['active']['seconds'] = time.monotonic()-started
                maps=Path(f'/proc/{child.pid}/maps')
                if maps.exists():
                    content=maps.read_text()
                    if 'libgaussian_mapper.so' in content:
                        assert str(ROOT/'runtime/lib/libgaussian_mapper.so') in content, '加载了非本轮Mapper'
                        (ROOT/'reports'/f'runtime_maps_{name}.txt').write_text(content)
                if output and (output/'real_lod_steps.csv').exists():
                    with (output/'real_lod_steps.csv').open('rb') as stream:
                        stream.seek(max(0,(output/'real_lod_steps.csv').stat().st_size-2048))
                        lines = stream.read().decode().splitlines()
                    state['active']['step'] = next((int(line.split(',')[0]) for line in reversed(lines)
                                                    if line.split(',')[0].isdigit()),0)
                save(state); time.sleep(3)
        result = dict(name=name, command=command, returncode=child.returncode, seconds=time.monotonic()-started)
        if child.returncode:
            state.update(state='failed',failure=result);save(state)
            raise RuntimeError(f'{name} failed; inspect logs/{name}.log, no automatic retry')
        return result

    tests = [('li_'+mode,[str(ROOT/'runtime/bin/local_importance_test'),mode,str(ROOT/'runs'/('test_li_'+mode))])
             for mode in ['unit','collection','on','off']]
    tests += [('legacy_'+mode,[str(ROOT/'runtime/bin/view_budget_training_test'),mode,str(ROOT/'runs'/('test_legacy_'+mode))])
              for mode in ['sampling','joint','empty','gradient','health']]
    tests += [('global_lod',[str(ROOT/'runtime/bin/real_lod_test'),str(ROOT/'runs/test_global_lod')])]
    for name, command in tests:
        state['tests'].append(execute(name,command));save(state)
    original = json.loads((OLD/'configs/manifest_joint_full.json').read_text())
    config = Path(original['selector_config']).read_text()
    assert 'ViewBudget.li_weight' not in config
    for name, weight in [('no_li',0),('li01',.1)]:
        selector = ROOT/'configs'/f'{name}.yaml'; selector.write_text(config+f'\nViewBudget.li_weight: {weight}\n')
        value = dict(original)
        for key in ['camera_config','mapping_config']:
            value[key] = str(ROOT/'configs'/Path(original[key]).name)
        value['selector_config'] = str(selector)
        value['output'] = str(ROOT/'runs'/f'smoke_{name}')
        value['input_sha256'] = {key:sha(value[key]) for key in original['input_sha256']}
        manifest = ROOT/'configs'/f'smoke_{name}.json';manifest.write_text(json.dumps(value,indent=2))
        state['smoke'][name] = execute('smoke_'+name,[str(ROOT/'runtime/bin/lod_from_full'),str(manifest),'12'],Path(value['output']))
        verified = json.loads((Path(value['output'])/'training_verified.json').read_text()); assert verified['passed']
    # CPU checks use PyTorch ABI0 only for reading, never to compile Mapper ABI1.
    import torch
    def tensors(folder):
        return torch.jit.load(str(folder/'step0_selector.pt'),map_location='cpu').state_dict()
    a,b = [ROOT/'runs'/f'smoke_{name}' for name in ['no_li','li01']]
    ta,tb = tensors(a),tensors(b)
    assert set(ta)==set(tb) and all(torch.equal(ta[k],tb[k]) for k in ta)
    assert sha(a/'step0_master.ply')==sha(b/'step0_master.ply')
    assert (a/'training_keyframes.csv').read_bytes()==(b/'training_keyframes.csv').read_bytes()
    state['initial_parity'] = {'passed':True,'tensors':len(ta),'master_sha256':sha(a/'step0_master.ply')}
    state['state'] = 'training';save(state)
    for name in ['no_li','li01']:
        assert shutil.disk_usage(ROOT).free>5*1024**3
        value = json.loads((ROOT/'configs'/f'smoke_{name}.json').read_text())
        output = ROOT/'runs'/f'full_{name}';assert not output.exists()
        value['output'] = str(output)
        manifest = ROOT/'configs'/f'full_{name}.json';manifest.write_text(json.dumps(value,indent=2))
        result = execute('full_'+name,[str(ROOT/'runtime/bin/lod_from_full'),str(manifest)],output)
        verified = json.loads((output/'training_verified.json').read_text())
        assert verified['passed'] and verified['steps']==11580 and verified['topology_unchanged']
        fine = json.loads((output/'23560_shutdown/ply/real_lod_state.json').read_text())
        assert fine['optimizer_commits']==fine['master_updates']==fine['selector_updates']==11580
        assert fine['view_budget_training_renders']==fine['full_training_renders']==11580
        assert fine['view_budget_li_full_renders']==(0 if name=='no_li' else 11580)
        shutil.copyfile(Path(value['full_ply']).parent.parent/'CameraTrajectory_TUM.txt',output/'CameraTrajectory_TUM.txt')
        state['groups'][name] = dict(result,verified=verified,checkpoint=str(output/'23560_shutdown/ply'))
        save(state)
    a,b = [ROOT/'runs'/f'full_{name}' for name in ['no_li','li01']]
    assert (a/'training_keyframes.csv').read_bytes()==(b/'training_keyframes.csv').read_bytes()
    ta,tb=tensors(a),tensors(b);assert all(torch.equal(ta[k],tb[k]) for k in ta)
    assert sha(a/'step0_master.ply')==sha(b/'step0_master.ply')
    state['state'] = 'training_complete';state['active']=None;save(state)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        if STATE.exists():
            state=json.loads(STATE.read_text());state.update(state='failed',error=repr(error));save(state)
        raise
