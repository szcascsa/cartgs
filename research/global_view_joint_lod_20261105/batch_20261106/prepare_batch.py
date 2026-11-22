"""为32组最佳Full模型复制隔离runtime，使用已验收的新联合Fine。"""
from pathlib import Path
import copy
import hashlib
import json
import os
import shutil
import subprocess

R = Path(__file__).resolve().parent
JOINT = R.parent/'global_view_joint_lod_20261105'
OLD = R.parent/'all_best_full_lod_no_frustum_20261104'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (R/'PREPARED.json').exists()
    assert shutil.disk_usage(R).free > 5*2**30
    for name in ['configs', 'reports', 'runs', 'logs', 'evaluation', 'validation']:
        (R/name).mkdir(exist_ok=True)
    if not (R/'source').exists():
        shutil.copytree(JOINT/'source', R/'source', symlinks=True, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copytree(JOINT/'runtime', R/'runtime', symlinks=True)
        shutil.copytree(JOINT/'evaluation/python', R/'source/python', dirs_exist_ok=True,
                        symlinks=True, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copytree(OLD/'evaluation_runtime', R/'evaluation_runtime', symlinks=True)
    else:
        assert not any((R/'runs').iterdir()) and not any((R/'validation').iterdir())
        for rel in ['runtime/bin/lod_from_full','runtime/lib/libgaussian_mapper.so','source/src/real_lod.cpp']:
            assert sha(R/rel)==sha(JOINT/rel), '未完成准备现场的已复制代码身份不匹配'
    # Python全轨迹入口已修正Stereo评价域；只在本批副本加局部预算适配。
    for name in ['pipeline.py','oom_retry.py','lod_eval_entry.py','verify_initialization.py']:
        shutil.copyfile(OLD/name, R/name)
    shutil.copyfile(OLD/'configs/vector_stereo_native_maps.yml.gz', R/'configs/vector_stereo_native_maps.yml.gz')
    # 使用真正完成fr1_xyz实验的启动配置，避免把源码中的四档模板当成六档实验配置。
    config = Path(json.loads((JOINT/'configs/manifest_joint_full.json').read_text())['selector_config'])
    text = config.read_text()
    for expected in ['ViewBudget.enabled: 1','ViewBudget.loss_weight: 1.0','ViewBudget.global_control: 0',
                     'RealLoD.full_loss_weight: 1.0','Selector.anchor_frustum_culling: 0',
                     'Selector.anchor_only_pruning: 1','Selector.target_ratios: [0.01, 0.05, 0.1, 0.3, 0.5, 0.7]']:
        assert expected in text, expected
    (R/'configs/joint.yaml').write_text(text)
    jobs = copy.deepcopy(json.loads((OLD/'SELECTED_BEST_FULL.json').read_text()))
    assert len(jobs) == 32
    protected = {'inputs': {}, 'production': {}}
    for j in jobs:
        j.update(state='pending', evaluations=[])
        for key in ['attempt_history','trained_model','process','attempt','result','error','failure']:
            j.pop(key, None)
        full = Path(j['full_job']['result']['checkpoint']).resolve()
        assert any(str(full).startswith('/home/prmi/zwj/realLOD/saved_experiments/'+n+'/') for n in
                   ['multi_dataset_full_tail_prune1_seed0_20261103','tum_full_tail_prune_20261103'])
        stage = json.loads((full/'real_lod_state.json').read_text())
        assert stage['lod_steps']==0 and stage['online_full_steps']==j['target_steps']
        j['full_job']['result']['checkpoint'] = str(full)
        j['full_evaluation']['model'] = str(Path(j['full_evaluation']['model']).resolve())
        j['camera_config'] = str(Path(j['camera_config']).resolve())
        j['mapping_config'] = str(Path(j['mapping_config']).resolve())
        cloud, = (full/'point_cloud').glob('iteration_*/point_cloud.ply')
        for p in [cloud, *cloud.parent.glob('*.pt'), *cloud.parent.glob('*.json'), full/'selector.pt',
                  full/'selector_config.yaml',full/'cameras.json',full/'real_lod_state.json',
                  Path(j['camera_config']),Path(j['mapping_config'])]:
            assert p.is_file(), p
            protected['inputs'][str(p)] = sha(p)
    # 不改变Full赢家和双目矫正后的选模口径；缺文件时停下而非偷偷换一遍。
    for name, value in json.loads((JOINT/'reports/PRODUCTION_BEFORE.json').read_text()).items():
        p = Path(name)
        if p.exists(): protected['production'][name] = sha(p)
    (R/'PROTECTED_BEFORE.json').write_text(json.dumps(protected, indent=2))
    (R/'SELECTED_BEST_FULL.json').write_text(json.dumps(jobs, ensure_ascii=False, indent=2))
    state = dict(phase='prepared',jobs=jobs,validated_modes=[],protocol=dict(seed=0,
        train_ratios=[.01,.05,.1,.3,.5,.7],unseen_ratios=[.03,.075,.2,.4,.6],
        objective='Full + global LoD + local LoD; 1:1:1; one shared Adam',
        local_pool='visible nominal Anchor members; training Gumbel-ST, no Top-K',
        evaluation_scopes=['global','view_ratio'],full_supervision=True,master_updates=True,
        topology_frozen=True,optimizer_reset=True,fine_steps='selected Full online Mapping count, excluding tail',
        saved_images=0,max_oom_attempts=3))
    (R/'LOD_PIPELINE.json').write_text(json.dumps(state, ensure_ascii=False, indent=2))
    env=dict(os.environ,LD_LIBRARY_PATH=str(R/'runtime/lib'))
    linkage=subprocess.check_output(['ldd',str(R/'runtime/bin/lod_from_full')],env=env,text=True)
    assert any('libgaussian_mapper.so =>' in line and str(R/'runtime/lib') in line for line in linkage.splitlines())
    (R/'reports/runtime_linkage.txt').write_text(linkage)
    ready=dict(passed=True,groups=32,total_fine_steps=sum(j['target_steps'] for j in jobs),
        binary_sha256=sha(R/'runtime/bin/lod_from_full'),mapper_sha256=sha(R/'runtime/lib/libgaussian_mapper.so'),
        config_sha256=sha(R/'configs/joint.yaml'))
    (R/'PREPARED.json').write_text(json.dumps(ready,indent=2));print(json.dumps(ready),flush=True)


if __name__=='__main__': main()
