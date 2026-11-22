"""冻结弱尾阈值2/255与溢出前缀两个独立候选，先边界再完整保留集双轮复验。"""
from pathlib import Path
import hashlib
import json
import subprocess
import time

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    previous = json.loads((HERE / 'reports/ADDITIONAL_CONTINUED_CONTROLLER.json').read_text())
    assert previous['state'] == 'complete', previous['state']
    latest = json.loads((HERE / 'reports/REMAINING_RESEARCH_20261115_CONTROLLER.json').read_text())
    assert latest['state'] == 'complete', latest['state']
    root = HERE / 'reports'
    frozen_path = root / 'FROZEN_TAIL_AND_OVERFLOW_20261115.json'
    assert not frozen_path.exists()
    frozen = dict(time=time.time(), development=['Replica/office2/rgbd', 'Replica/room0/rgbd'],
        alpha_floor='2/255', alpha_mode=203, overflow_mode=182,
        selection_policy='alpha2优先；若质量或可重复性能不通过则回退182。overflow仅作为独立候选，不逐场景配参。',
        no_holdout_tuning=True, no_input_or_K_changes=True, variants={})
    for name in ['tail_cutoff', 'overflow_prefix']:
        source = HERE / ('source_' + name)
        binary = next((HERE / ('build_' + name)).glob('*.so'))
        frozen['variants'][name] = dict(binary=str(binary), binary_sha256=sha(binary),
            sources={str(path.relative_to(HERE)): sha(path) for path in source.rglob('*')
                     if path.is_file() and path.suffix in ['.cu', '.cpp', '.cuh', '.h']})
    frozen_path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2))
    jobs = []
    for label, modes in [('tail_cutoff', '203'), ('overflow_prefix', '182')]:
        jobs.append(dict(name=f'new_{label}_edges', measure=False,
            command=['{python}', '-u', '-B', f'{label}_edges.py', '--output', '{job}', '--modes', modes]))
    for repeat in [1, 2]:
        for label, groups in [('formal', 9), ('nested', 16)]:
            for group in (range(groups) if repeat == 1 else reversed(range(groups))):
                jobs.append(dict(name=f'new_tail_validation_{label}_g{group}_r{repeat}', measure=True,
                    command=['{python}', '-u', '-B', f'tail_cutoff_{label}.py', '--output', '{job}',
                             '--group', str(group), '--round', str(repeat)]))
    for label, modes in [('tail_cutoff', '203'), ('overflow_prefix', '182')]:
        for tool in ['memcheck', 'initcheck', 'synccheck']:
            jobs.append(dict(name=f'new_{label}_{tool}', measure=False,
                command=['/usr/local/cuda-11.8/bin/compute-sanitizer', '--tool', tool, '--error-exitcode', '99',
                         '--print-limit', '10', '{python}', '-u', '-B', f'{label}_edges.py', '--output', '{job}',
                         '--modes', modes, '--sanitizer']))
    path = root / 'TAIL_VALIDATION_20261115_PLAN.json'
    assert not path.exists()
    path.write_text(json.dumps(dict(name='TAIL_VALIDATION_20261115', jobs=jobs), indent=2))
    with (HERE / 'logs/tail_validation_20261115_controller.log').open('x') as log:
        child = subprocess.Popen(['/home/prmi/anaconda3/envs/zwj/bin/python', '-u', '-B', 'run_stage.py', str(path)],
            cwd=HERE, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print('TAIL_VALIDATION_LAUNCHED', child.pid, len(jobs), flush=True)


if __name__ == '__main__':
    main()
