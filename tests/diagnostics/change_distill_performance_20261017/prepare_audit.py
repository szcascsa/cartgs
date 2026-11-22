"""冻结本地源码、配置和已有证据。只写本审计目录。"""
from pathlib import Path
import csv
import hashlib
import json
import re
import shutil
import subprocess

D = Path(__file__).resolve().parent
REPO = D.parents[2]
PRIOR = REPO.parent / 'change-distill-ablation-20261017'
RAW = D / 'raw'
RAW.mkdir(exist_ok=True)
SOURCE = D / 'source'
FILES = [
    'include/change_management.h', 'include/frozen_teacher_target_cache.h',
    'include/gaussian_transform_ops.h', 'include/quaternion_utils.h',
    'include/model_snapshot.h', 'include/transform_field.h',
    'include/hexplane_field.h', 'include/aabb.h', 'include/selector_cuda.h',
    'include/transform_field_cuda.h', 'include/transform_field_cuda_optimizer.h',
    'include/protection_profiler.h', 'include/elastic_entry.h', 'include/selector_handoff.h',
    'improvements/selector/gumbel_network.h',
    'src/change_distill.cpp', 'src/change_management.cpp', 'src/transform_field.cpp',
    'src/hexplane_field.cpp', 'src/selector_cuda.cpp', 'src/selector_cuda.cu',
    'src/transform_field_cuda.cpp', 'src/transform_field_cuda.cu',
    'src/transform_field_cuda_neural.cu', 'src/transform_field_cuda_regularization.cu',
    'src/transform_field_cuda_optimizer.cpp', 'src/transform_field_cuda_optimizer.cu',
    'src/protection_profiler.cpp',
    'src/gaussian_mapper.cpp', 'cfg/selector/fast.yaml',
    'cfg/selector/selector_config.yaml', 'CMakeLists.txt',
]
manifest = {}
for relative in FILES:
    src = REPO / relative
    if not src.exists():
        continue
    dst = SOURCE / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = src.read_bytes()
    dst.write_bytes(data)
    manifest[relative] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
(RAW / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
for args, name in [(['git', 'status', '--short'], 'git_status_before.txt'),
                   (['git', 'diff', '--binary'], 'working_tree_before.diff'),
                   (['git', 'log', '-1', '--format=%H %s'], 'git_head.txt')]:
    (RAW / name).write_bytes(subprocess.check_output(args, cwd=REPO))
event_path = PRIOR / 'events_supplement/local_loop/event_6207_1/event.json'
shutil.copyfile(event_path, RAW / 'event_6207.json')
event = json.loads(event_path.read_text())
summary = {k: v for k, v in event.items() if k not in ('views', 'groups')}
summary['correction_group_count'] = len(event['groups'])
summary['snapshot_local_available'] = (event_path.parent / 'snapshot.pt').exists()
(RAW / 'event_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
configs = {}
for name in ('fast.yaml', 'selector_config.yaml'):
    values = {}
    for line in (SOURCE / 'cfg/selector' / name).read_text(encoding='utf-8').splitlines():
        if ':' in line and not line.lstrip().startswith(('#', '%')):
            key, value = line.split(':', 1)
            if key.startswith(('ChangeDistill.', 'Selector.target_ratios', 'Selector.backend',
                               'TransformField.backend', 'TransformField.feature_dim',
                               'TransformField.multires', 'TransformField.hidden_dim')):
                values[key] = value.strip()
    shared = int(values['ChangeDistill.epochs'])
    values['effective_selector_epochs'] = int(values.get('ChangeDistill.selector_epochs', shared))
    values['effective_tf_epochs'] = int(values.get('ChangeDistill.tf_epochs', shared))
    configs[name] = values
(RAW / 'current_configs.json').write_text(json.dumps(configs, indent=2), encoding='utf-8')
timings = []
base = PRIOR / 'paired_results/local_loop_event_6207_1/seed0'
for path in sorted(base.glob('*timing*/runtime.json')):
    row = json.loads(path.read_text())
    row['source'] = str(path.relative_to(PRIOR)).replace('\\', '/')
    row['run'] = path.parent.name
    log = path.with_name('optimize.log').read_text(errors='replace')
    for module in ('SELECTOR', 'TF'):
        matches = re.findall(r'\[' + module + r' CD\].*?time_ms=([0-9.]+)', log)
        if matches:
            row[module.lower() + '_reported_ms'] = float(matches[-1])
    timings.append(row)
(RAW / 'previous_timing_raw.json').write_text(json.dumps(timings, indent=2), encoding='utf-8')
if timings:
    fields = sorted(set().union(*(x.keys() for x in timings)))
    with (RAW / 'previous_timing_raw.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(timings)
rows = []
pattern = re.compile(r'\.cpu\(|\.cuda\(|\.to\(|\.item\s*<|data_ptr|TensorAccessor|std::vector|'
                     r'torch::(?:zeros|ones|empty|full|cat|stack|nonzero)|'
                     r'\.(?:clone|contiguous|index_select|index_copy|index_put_|masked_select|gather)\(')
for name in manifest:
    if not name.endswith(('.h', '.cpp', '.cu')):
        continue
    for line_no, line in enumerate((SOURCE / name).read_text(encoding='utf-8').splitlines(), 1):
        if pattern.search(line):
            rows.append({'file': name, 'line': line_no, 'code': line.strip()})
with (RAW / 'source_sync_allocation_scan.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, ['file', 'line', 'code'])
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({'frozen_files': len(manifest), 'previous_runs': len(timings),
                  'event': summary, 'configs': configs}, ensure_ascii=False, indent=2))
