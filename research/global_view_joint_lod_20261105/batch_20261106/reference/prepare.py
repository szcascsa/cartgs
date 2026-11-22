"""复制已验收runtime，建立32组无视锥LoD独立队列，不修改生产。"""
from pathlib import Path
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess

from select_full import select_winners

R = Path(__file__).resolve().parent
B = R.parent
PREVIOUS = B / 'best_full_lod_20261104'
REFERENCE = B / 'corridors_no_frustum_full_fine_20261104'
PATH_FIXED = B / 'fr1xyz_continuous_vs_checkpoint_20261104'
ARCHIVE = Path('/home/prmi/zwj/realLOD/saved_experiments')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (R / 'PREPARED.json').exists(), '禁止覆盖已有实验'
    assert not (R / 'source').exists(), '已有现场需核查'
    assert shutil.disk_usage(R).free > 5 * 1024**3
    for part in ['source', 'runtime', 'evaluation_runtime']:
        shutil.copytree(REFERENCE / part, R / part, symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    # 唯一入口差别是相对img_name在明确dataset_root下解析，已在fr1验收。
    assert sha(PATH_FIXED / 'runtime/lib/libgaussian_mapper.so') == sha(R / 'runtime/lib/libgaussian_mapper.so')
    shutil.copy2(PATH_FIXED / 'runtime/bin/lod_from_full', R / 'runtime/bin/lod_from_full')
    shutil.copy2(PATH_FIXED / 'lod_from_full.cpp', R / 'lod_from_full.cpp')
    for name in ['pipeline.py', 'oom_retry.py', 'lod_eval_entry.py', 'verify_initialization.py']:
        shutil.copy2(PREVIOUS / name, R / name)
    for name in ['configs', 'reports', 'logs', 'runs', 'evaluation', 'validation']:
        (R / name).mkdir(exist_ok=True)
    shutil.copy2(REFERENCE / 'configs/train_no_frustum.yaml', R / 'configs/train_no_frustum.yaml')
    shutil.copy2(REFERENCE / 'configs/vector_stereo_native_maps.yml.gz', R / 'configs/vector_stereo_native_maps.yml.gz')

    training, evaluation = [], []
    sources = {}
    for name in ['tum_full_tail_prune_20261103', 'multi_dataset_full_tail_prune1_seed0_20261103']:
        root = ARCHIVE / name
        q = read(root / 'QUEUE.json')
        e = read(root / 'TRAJECTORY_EVALUATION.json')['jobs']
        training.extend(q); evaluation.extend(e)
        for rel in ['QUEUE.json', 'TRAJECTORY_EVALUATION.json', 'SCENE_INVENTORY.json', 'configs/selector_prune1.yaml']:
            sources[str(root / rel)] = sha(root / rel)
        old = (root / 'configs/selector_prune1.yaml').read_text()
        old, count = re.subn(r'^Selector.target_ratios:.*$',
                            'Selector.target_ratios: [0.01, 0.05, 0.1, 0.3, 0.5, 0.7]', old, flags=re.M)
        assert count == 1
        old, count = re.subn(r'^Selector.anchor_frustum_culling: 1$',
                            'Selector.anchor_frustum_culling: 0', old, flags=re.M)
        assert count == 1
        assert old == (R / 'configs/train_no_frustum.yaml').read_text(), '除ratio节点和视锥之外配置不匹配'
    correction = B / 'lod_health_repair_20261104/reports/STEREO_CORRECTED_METRICS.csv'
    with correction.open() as stream:
        jobs = select_winners(training, evaluation, list(csv.DictReader(stream)))
    assert len(jobs) == 32
    sources[str(correction)] = sha(correction)
    for job in jobs:
        full = Path(job['full_job']['result']['checkpoint']).resolve()
        assert full.is_relative_to(ARCHIVE)
        job['full_job']['result']['checkpoint'] = str(full)
        job['full_evaluation']['model'] = str(Path(job['full_evaluation']['model']).resolve())
        stage = read(full / 'real_lod_state.json')
        assert stage['lod_steps'] == 0 and stage['online_full_steps'] == job['target_steps']
        camera, mapping = job['full_job']['command'][2:4]
        job['camera_config'] = str(Path(camera).resolve())
        job['mapping_config'] = str(Path(mapping).resolve())
        views = read(full / 'cameras.json')
        assert views
        data = Path(job['full_evaluation']['data'])
        for view in views:
            image = Path(view['img_name'])
            assert (image if image.is_absolute() else data / image).is_file(), (job['id'], image)
        point_cloud, = (full / 'point_cloud').glob('iteration_*/point_cloud.ply')
        protected = [point_cloud, *point_cloud.parent.glob('*.json'), *point_cloud.parent.glob('*.pt'),
                     full / 'selector.pt', full / 'selector_config.yaml', full / 'real_lod_state.json',
                     full / 'cameras.json', Path(job['camera_config']), Path(job['mapping_config'])]
        for path in protected:
            assert path.is_file(), str(path)
            sources[str(path)] = sha(path)
    production = {}
    for root in [Path('/home/prmi/zwj/realLOD'), Path('/home/prmi/zwj/mycartgs')]:
        for rel in ['src/gaussian_mapper.cpp', 'src/real_lod.cpp', 'include/gaussian_mapper.h',
                    'cfg/selector/selector_config.yaml', 'lib/libgaussian_mapper.so', 'bin/tum_rgbd']:
            path = root / rel
            if path.is_file(): production[str(path)] = sha(path)
    before = dict(inputs=sources, production=production)
    (R / 'PROTECTED_BEFORE.json').write_text(json.dumps(before, indent=2))
    linkage = subprocess.check_output(['ldd', str(R / 'runtime/bin/lod_from_full')],
        env=dict(os.environ, LD_LIBRARY_PATH=str(R / 'runtime/lib')), text=True)
    assert any('libgaussian_mapper.so =>' in line and str(R / 'runtime/lib') in line for line in linkage.splitlines())
    (R / 'reports/runtime_linkage.txt').write_text(linkage)
    state = dict(phase='prepared', jobs=jobs, validated_modes=[], protocol=dict(
        seed=0, anchor_only_pruning=1, anchor_frustum_culling=0,
        train_ratios=[.01, .05, .1, .3, .5, .7], unseen_ratios=[.03, .075, .2, .4, .6],
        master_updates=True, full_plus_rotating_lod=True, topology_frozen=True,
        fine_steps='所选Full运行online Mapping步数，不含tail', optimizer_reset=True,
        primary_evaluation='全轨迹；全图候选；Stereo使用rectified GT；Mono校正尺度',
        saved_images=0, health_guard=True, max_oom_attempts=3))
    (R / 'SELECTED_BEST_FULL.json').write_text(json.dumps(jobs, ensure_ascii=False, indent=2))
    (R / 'LOD_PIPELINE.json').write_text(json.dumps(state, ensure_ascii=False, indent=2))
    ready = dict(passed=True, groups=len(jobs), total_fine_steps=sum(j['target_steps'] for j in jobs),
                 binary_sha256=sha(R / 'runtime/bin/lod_from_full'),
                 mapper_sha256=sha(R / 'runtime/lib/libgaussian_mapper.so'),
                 reused_path_fixed_entry=True, reused_health_guard=True)
    (R / 'PREPARED.json').write_text(json.dumps(ready, indent=2))
    print(json.dumps(ready), flush=True)


if __name__ == '__main__':
    main()
