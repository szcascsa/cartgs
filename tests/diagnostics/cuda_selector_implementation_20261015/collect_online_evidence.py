"""收集完成的在线 A/B 原始指标、日志摘录和哈希；不复制模型或改动生产目录。"""
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

dest = Path(__file__).resolve().parent
root = dest.parents[2]
online = root/'results/selector_online_ab_final_20261015'
state = json.loads((online/'status.json').read_text())
assert state.get('complete') and not state.get('error')
assert len(state['runs']) == 24 and all(r.get('complete') for r in state['runs'])
assert all(r['train']['exit'] == r['evaluation']['exit'] == 0 for r in state['runs'])
evidence = dest/'online_evidence'
evidence.mkdir(exist_ok=True)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


files = {}


def record(path):
    files[str(path.relative_to(root))] = dict(bytes=path.stat().st_size, sha256=digest(path))


for name in ('status.json', 'source_fast.yaml', 'legacy.yaml', 'cuda.yaml'):
    shutil.copy2(online/name, evidence/name)
    record(online/name)
assert digest(online/'source_fast.yaml') == state['source_config_sha256']
configs = [(online/f'{backend}.yaml').read_text().replace(f'Selector.backend: "{backend}"', '')
           for backend in ('legacy', 'cuda')]
assert configs[0] == configs[1], 'A/B 配置除 backend 外必须相同'
for run in state['runs']:
    source = online/run['name']
    target = evidence/run['name']
    target.mkdir(parents=True, exist_ok=True)
    for name in ('train.log', 'evaluation.log'):
        record(source/name)
    lines = (source/'train.log').read_text(errors='replace').splitlines()
    markers = ('[Mapping PERF FINAL]', '[CUDA TF ONLINE]', '[Continuous', '[UltrafastRotation]')
    training = [line for line in lines if '[UltrafastRotation]' in line and 'training_ms=' in line]
    final = [line for line in lines if any(marker in line for marker in markers[:3])]
    (target/'train_excerpt.log').write_text('\n'.join(training[:8]+final+lines[-20:])+'\n')
    metrics = list((source/'evaluation').rglob('eval.txt'))
    assert len(metrics) == 5
    trajectories = []
    for metric in metrics:
        scores = dict(re.findall(r'^(psnr|ssim|lpips):\s*([-+0-9.eE]+)', metric.read_text(), re.M))
        assert len(scores) == 3 and all(math.isfinite(float(v)) for v in scores.values()), metric
        for name in ('eval.txt', 'metrics_traj.txt', 'selector_stats.json'):
            path = metric.parent/name
            assert path.exists(), path
            output = target/'evaluation'/path.parent.name/name
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, output)
            record(path)
        trajectories.append((metric.parent/'metrics_traj.txt').read_text())
    assert len(set(trajectories)) == 1, '同一次训练的五个评估 ratio 必须使用同一条 trajectory'
    checkpoints = list(source.glob('*_shutdown/ply'))
    assert len(checkpoints) == 1
    for pattern in ('selector.pt', 'selector_config.yaml', 'point_cloud/iteration_*/point_cloud.ply',
                    'point_cloud/iteration_*/selector_metadata.ply'):
        paths = list(checkpoints[0].glob(pattern))
        assert len(paths) == 1, (source, pattern, paths)
        record(paths[0])
    shutil.copy2(checkpoints[0]/'selector_config.yaml', target/'saved_selector_config.yaml')

runtime = {name: digest(root/name) for name in ('bin/tum_rgbd_cuda_tf_online', 'lib/libgaussian_mapper.so')}
assert runtime['bin/tum_rgbd_cuda_tf_online'] == state['binary_sha256']
assert runtime['lib/libgaussian_mapper.so'] == state['mapper_library_sha256']
before = json.loads((dest/'round2_environment_before.json').read_text())['production_files']
production = Path('/home/prmi/zwj/mycartgs')
after = {name: digest(production/name) for name in before}
assert before == after, '生产目录哈希发生改变'
default_legacy = 'std::string selector_backend_ = "legacy"' in (root/'include/gaussian_mapper.h').read_text()
build_gate_off = 'isolated A/B tests" OFF)' in (root/'CMakeLists.txt').read_text()
assert default_legacy and build_gate_off
source_files = ['CMakeLists.txt', 'cfg/selector/selector_config.yaml', 'improvements/selector/gumbel_network.h',
                'include/gaussian_mapper.h', 'include/selector_cuda.h', 'src/selector_cuda.cpp',
                'src/selector_cuda.cu', 'src/gaussian_mapper.cpp', 'src/change_distill.cpp',
                'src/drift_stability.cpp', 'python/run.py', 'python/selector.py',
                'python/selector_cuda_backend.py', 'python/selector_cuda_bindings.cpp']
summary = dict(production_accepted=False, completed_runs=24, quality_metric_sets=120,
               same_config_except_backend=True, stable_online_binary_and_library=True,
               runtime_hashes=runtime, production_files_unchanged=True, production_files=after,
               default_legacy=default_legacy, experimental_build_default_off=build_gate_off,
               delivery_source_hashes={name: digest(root/name) for name in source_files},
               gpu_processes=subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                             '--format=csv,noheader'], capture_output=True, text=True).stdout,
               files=files)
(dest/'online_evidence_manifest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
shutil.copy2(online/'online_summary.json', dest/'online_summary.json')
print(json.dumps({k: v for k, v in summary.items() if k not in ('files', 'production_files')}, ensure_ascii=False))
