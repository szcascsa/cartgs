"""匹配的两组蒸馏，失败即保留现场并终止。"""
from pathlib import Path
import hashlib
import json
import subprocess
import time


ROOT = Path(__file__).resolve().parents[3]
PYTHON = '/home/prmi/anaconda3/envs/zwj/bin/python'
HERE = Path(__file__).resolve().parent
BASELINE = '/home/prmi/zwj/experiments/compensator_distillation_20261108/models/fr1_student32_image_calibration/best.pt'
TEACHER = '/home/prmi/zwj/experiments/view_conditioned_li_20261106/runs/full_both_li/23560_shutdown/ply'
DATASET = '/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz'


def save(value):
    path = ROOT/'reports/PAIRED_RUNS_STATUS.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    test = subprocess.run([PYTHON, '-u', str(HERE/'test_objective.py')], cwd=ROOT)
    if test.returncode:
        save(dict(state='test_failed', returncode=test.returncode))
        raise SystemExit(test.returncode)
    for name, weight in [('control', 1.), ('residual001', .01)]:
        output = ROOT/'models'/name
        command = [PYTHON, '-u', str(HERE/'run.py'), 'train', TEACHER, '--dataset', DATASET,
            '--output', str(output), '--initialize', BASELINE, '--steps', '2400', '--lr', '.0002',
            '--image-every', '1', '--image-weight', '200', '--residual-weight', str(weight),
            '--validation-metric', 'image_mse', '--validate-every', '300', '--seed', '0']
        with (ROOT/'logs'/f'{name}.log').open('x') as log:
            p = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            state = dict(state='running', experiment=name, pid=p.pid,
                start_ticks=int(Path(f'/proc/{p.pid}/stat').read_text().rpartition(')')[2].split()[19]),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                started=time.time(), command=command,
                initial_sha256=hashlib.sha256(Path(BASELINE).read_bytes()).hexdigest())
            save(state)
            code = p.wait()
            state.update(state='complete' if code == 0 else 'failed', returncode=code, ended=time.time())
            save(state)
            if code:
                raise SystemExit(code)
    save(dict(state='all_complete', experiments=['control', 'residual001'], images_saved=0))
    print('PAIRED_RUNS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
