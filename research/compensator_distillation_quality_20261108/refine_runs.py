"""连续 Ratio 采样的匹配训练与独立最终验收。"""
from pathlib import Path
import hashlib
import json
import subprocess
import time


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PYTHON = '/home/prmi/anaconda3/envs/zwj/bin/python'


def main():
    assert json.loads((ROOT/'reports/DELIVERY_TESTS.json').read_text())['complete']
    protocol = json.loads((ROOT/'reports/REFINEMENT_PROTOCOL.json').read_text())
    initial = ROOT/protocol['initial_student']
    assert hashlib.sha256(initial.read_bytes()).hexdigest() == protocol['initial_sha256']
    state = dict(tasks=[], started=time.time(), protocol=protocol)

    def run(name, arguments):
        with (ROOT/'logs'/f'{name}.log').open('x') as log:
            p = subprocess.Popen([PYTHON, '-u', *arguments], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            task = dict(name=name, pid=p.pid, started=time.time(),
                start_ticks=int(Path(f'/proc/{p.pid}/stat').read_text().rpartition(')')[2].split()[19]),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), command=arguments)
            state['tasks'].append(task)
            (ROOT/'reports/REFINEMENT_STATUS.json').write_text(json.dumps(state, indent=2, ensure_ascii=False))
            code = p.wait()
            task.update(returncode=code, ended=time.time())
            (ROOT/'reports/REFINEMENT_STATUS.json').write_text(json.dumps(state, indent=2, ensure_ascii=False))
            print(name, code, flush=True)
            if code:
                raise SystemExit(code)

    run('ratio_sampling_test', [str(HERE/'test_ratio_sampling.py')])
    run('refinement_objective_test', [str(HERE/'test_objective.py')])
    for name, jitter in protocol['ratio_jitter'].items():
        run(name, [str(HERE/'run.py'), 'train',
            '/home/prmi/zwj/experiments/view_conditioned_li_20261106/runs/full_both_li/23560_shutdown/ply',
            '--dataset', '/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz',
            '--output', str(ROOT/'models'/name), '--initialize', str(initial),
            '--steps', str(protocol['steps_per_arm']), '--lr', str(protocol['lr']),
            '--image-every', '1', '--image-weight', '200', '--residual-weight', '.01',
            '--validation-metric', 'image_mse', '--validate-every', '300',
            '--validation-ratios', *[str(r) for r in protocol['validation_ratios']],
            '--ratio-jitter', str(jitter), '--paired-view-sampling'])
    import csv
    logs = {name:list(csv.DictReader((ROOT/'models'/name/'training.csv').open())) for name in protocol['ratio_jitter']}
    names = list(logs)
    assert len(logs[names[0]]) == len(logs[names[1]]) == protocol['steps_per_arm']
    assert [r['frame'] for r in logs[names[0]]] == [r['frame'] for r in logs[names[1]]]
    assert [r['local'] for r in logs[names[0]]] == [r['local'] for r in logs[names[1]]]
    for name, rows in logs.items():
        assert all(r['ratio'] == r['image_ratio'] for r in rows)
    (ROOT/'reports/REFINEMENT_EXPOSURE_CHECK.json').write_text(json.dumps(dict(passed=True,
        steps_per_arm=protocol['steps_per_arm'], frames_identical=True, modes_identical=True,
        image_ratio_matches_attribute_ratio=True), indent=2))
    run('final_quality', [str(HERE/'compare_quality.py'), str(ROOT/'reports/FINAL_COMPARISON_PLAN.json')])
    run('final_refinement_validation', [str(HERE/'final_validation.py')])
    run('perceptual_quality', [str(HERE/'perceptual_quality.py')])
    run('final_refinement_integrity', [str(HERE/'integrity.py')])
    state.update(complete=True, ended=time.time())
    (ROOT/'reports/REFINEMENT_STATUS.json').write_text(json.dumps(state, indent=2, ensure_ascii=False))
    print('REFINEMENT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
