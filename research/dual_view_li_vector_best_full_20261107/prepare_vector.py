"""建立VECtor独立队列，复用本批源码和二进制，保留原暂停状态。"""
from pathlib import Path
import collections
import copy
import json
import shutil
import time

B = Path('/home/prmi/zwj/realLOD/saved_experiments/dual_view_li_best_full_20261107')
R = Path(__file__).resolve().parent


def validate_jobs(jobs):
    expected = {(scene, mode) for scene in ['corner-slow', 'corridors-dolly', 'robot-normal']
                for mode in ['mono', 'stereo']}
    assert len(jobs) == 6 and {(j['scene'], j['mode']) for j in jobs} == expected
    assert all(j['dataset'] == 'VECtor' and j['id'] == f"VECtor/{j['scene']}/{j['mode']}" for j in jobs)
    for job in jobs:
        candidates = job['candidates']
        assert sorted(c['repeat'] for c in candidates) == [1, 2, 3]
        assert all(c['train_state'] == c['eval_state'] == 'complete' for c in candidates)
        winner = max(candidates, key=lambda c: (c['result']['trajectory_psnr'], -c['repeat']))
        assert winner['id'] == job['winner']
        assert winner['result']['trajectory_psnr'] == job['full_psnr']
        if job['mode'] == 'stereo':
            assert all('uncorrected_trajectory_psnr' in c['result'] and not c['result']['correct_scale'] for c in candidates)
        assert job['target_steps'] == job['full_job']['result']['online_steps'] > 0


def main():
    import sys
    sys.path.insert(0, str(B))
    import pipeline as p
    import supervisor

    assert R.name == 'dual_view_li_vector_best_full_20261107'
    assert not (R/'PREPARED.json').exists() and not (R/'LOD_PIPELINE.json').exists()
    assert shutil.disk_usage(R).free > 5*2**30
    original = p.read(B/'LOD_PIPELINE.json')
    assert original['phase'] == 'held' and (B/'HOLD').exists()
    assert all(not supervisor.alive(p.read(B/name)) for name in ['SUPERVISOR.json', 'LOD_CONTROLLER.json'])
    deferred = p.read(B/'DEFERRED_VECTOR_BY_USER.json')['jobs']
    validate_jobs(deferred)
    for name in ['configs', 'reports', 'logs', 'runs', 'evaluation', 'validation']:
        (R/name).mkdir(exist_ok=True)
    protected = p.read(B/'PROTECTED_BEFORE.json')
    protected['paused_original_batch'] = {str(B/name): p.sha(B/name)
        for name in ['LOD_PIPELINE.json', 'HOLD', 'DEFERRED_VECTOR_BY_USER.json', 'configs/dual_view_li.yaml']}
    assert all(p.sha(Path(name)) == digest for files in protected.values() for name, digest in files.items())
    sources = p.read(B/'reports/SOURCE_HASHES.json')
    assert all(p.sha(B/'source'/name) == digest for name, digest in sources.items())
    shutil.copytree(B/'source', R/'source', symlinks=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(B/'runtime', R/'runtime', symlinks=True)
    shutil.copytree(B/'evaluation_runtime', R/'evaluation_runtime', symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ['pipeline.py', 'oom_retry.py', 'lod_eval_entry.py', 'eval_joint_entry.py',
                 'verify_initialization.py', 'validate_startup.py', 'test_queue.py', 'supervisor.py']:
        shutil.copy2(B/name, R/name)
    runner = (B/'run_batch.py').read_text()
    old_scope = "assert len(state['jobs'])==26 and all(j['dataset']!='VECtor' for j in state['jobs'])"
    new_scope = "assert len(state['jobs'])==6 and all(j['dataset']=='VECtor' for j in state['jobs'])"
    assert runner.count(old_scope) == 1
    runner = runner.replace(old_scope, new_scope).replace(
        '26组双网络真实视角＋LI队列；VECtor暂不跑，持久化重试、只保留最终模型。',
        '6组VECtor双网络真实视角＋LI独立队列，持久化重试、只保留最终模型。')
    (R/'run_batch.py').write_text(runner)
    shutil.copy2(B/'configs/dual_view_li.yaml', R/'configs/dual_view_li.yaml')
    shutil.copy2(B/'configs/vector_stereo_native_maps.yml.gz', R/'configs/vector_stereo_native_maps.yml.gz')
    jobs = copy.deepcopy(deferred)
    for job in jobs:
        job.update(state='pending', evaluations=[], attempt_history=[], health_failures=0, oom_failures=0)
        for key in ['camera_config', 'mapping_config']:
            source = Path(job[key])
            destination = R/source.relative_to(B)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            assert p.sha(source) == p.sha(destination)
            job[key] = str(destination)
        full = Path(job['full_job']['result']['checkpoint'])
        assert p.read(full/'real_lod_state.json')['lod_steps'] == 0
        assert p.read(full/'real_lod_state.json')['online_full_steps'] == job['target_steps']
        assert Path(job['full_evaluation']['data']).is_dir()
        for camera in p.read(full/'cameras.json'):
            image = Path(camera['img_name'])
            if not image.is_absolute():
                image = Path(job['full_evaluation']['data'])/image
            assert image.is_file(), image
    build = p.read(B/'reports/BUILD_HASHES.json')
    runtime = {name: digest for name, digest in build.items() if name.startswith('runtime/')}
    assert len(runtime) > 0 and all(p.sha(R/name) == digest for name, digest in runtime.items())
    assert all(p.sha(R/'source'/name) == digest for name, digest in sources.items())
    for name in ['SOURCE_HASHES.json', 'SOURCE_PROVENANCE.json']:
        shutil.copy2(B/'reports'/name, R/'reports'/name)
    p.write(R/'reports/BUILD_HASHES.json', runtime)
    p.write(R/'PROTECTED_BEFORE.json', protected)
    p.write(R/'SELECTED_BEST_FULL.json', jobs)
    state = dict(phase='prepared', jobs=jobs, validated_modes=[], protocol=copy.deepcopy(original['protocol']))
    p.R = R
    p.save(state)
    p.write(R/'reports/BOOTSTRAP.json', dict(phase='build_complete', time=time.time(),
        runtime_reused_from=str(B), runtime_hashes_verified=len(runtime), independent_runtime=True,
        source_hashes_verified=len(sources)))
    p.write(R/'reports/USER_AUTHORIZATION.json', dict(
        authorized_unix=time.time(), request='用户要求利用当前服务器空间运行此前暂缓的VECtor场景',
        groups=6, modes=['mono', 'stereo'], existing_replica_tum_queue_stays_held=True,
        extra_health_retries=2, extra_oom_retries=2, saved_render_images=0))
    receipt = dict(passed=True, groups=6, total_fine_steps=sum(j['target_steps'] for j in jobs),
        counts=dict(collections.Counter(j['mode'] for j in jobs)), source_files=len(sources),
        runtime_hashes_verified=len(runtime), original_batch_state='held', root=str(R))
    p.write(R/'PREPARED.json', receipt)
    p.write(R/'reports/QUEUE_SOURCE_HASHES.json', {f.name: p.sha(f) for f in R.glob('*.py')})
    lines = ['# VECtor双网络真实视角＋LI实验协议', '',
        'corner-slow、corridors-dolly、robot-normal各运行Mono和双目，共6组。'
        '每组使用seed0三次中Full全轨迹PSNR最高的原Full起点，双目赢家按校正后左图域的指标选取。', '',
        '本轮复制此前双网络真实视角＋LI批次的527个源码文件及16个运行文件，运行文件SHA256相同。'
        '配置为LI=0.1、only anchor prune=1、Full＋全局LoD＋局部LoD联合反传后一次共享Adam。'
        'Master可更新，拓扑固定，Adam/RNG重新初始化，Fine步数等于原online Mapping步数。', '',
        '训练ratio为1%、5%、10%、30%、50%、70%。未见ratio为3%、8%、20%、40%、60%，仅评价。'
        '每组进行11档全局、11档名义Anchor候选池局部及Full的完整轨迹评价，共23项。'
        '输出PSNR、SSIM、LPIPS、ATE、Gaussian数量、实际K及逐帧指标，渲染帧图片保存数为0。', '',
        'Mono使用尺度对齐。双目使用原生OpenCV左图校正map与保存的相机内参，尺度对齐关闭，'
        '采用整个校正图像计算指标。每种入口先执行6步训练及Full、全局1%/3%/70%、局部1%验收。', '',
        'OOM和健康停止各最多额外重试2次，计数跨控制器重启保留。未知错误阻塞并保存现场。'
        '监督器每小时核查真实进程、日志、步数、磁盘和图片数。原Replica/TUM批次保持暂停。', '',
        '|场景／模式|Full起点PSNR|Fine步数|Gaussian数|评价帧数|', '|---|---:|---:|---:|---:|']
    for job in jobs:
        lines.append(f"|{job['id']}|{job['full_psnr']:.6f}|{job['target_steps']}|"
                     f"{job['full_job']['result']['gaussian_count']}|{job['full_evaluation']['result']['evaluated_frames']}|")
    (R/'reports/BATCH_PROTOCOL.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
