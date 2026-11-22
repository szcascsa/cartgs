"""串行筛选空间变化/exp2/辅助读回候选，并补充原C++入口及单实例显存对照。"""
from pathlib import Path
import json
import subprocess

HERE = Path(__file__).resolve().parent


def main():
    state = json.loads((HERE / 'reports/CANDIDATES_20261115_CONTROLLER.json').read_text())
    assert state['state'] == 'complete', state['state']
    jobs = []
    for label in ['pair_coarse', 'exp2_encoded', 'async_control']:
        jobs.append(dict(name=f'new_{label}_source', measure=False,
            command=['{python}', '-u', '-B', f'implement_{label}.py']))
        jobs.append(dict(name=f'new_{label}_build', measure=False,
            command=['{python}', '-u', '-B', f'{label}_runtime.py']))
        for repeat in [1, 2]:
            command = ['{python}', '-u', '-B', f'{label}_development.py', '--output', '{job}', '--round', str(repeat)]
            if label != 'async_control':
                command += ['--group', '0']
            jobs.append(dict(name=f'new_{label}_r{repeat}', measure=True, command=command))
    for repeat in [1, 2]:
        for group in (range(9) if repeat == 1 else reversed(range(9))):
            jobs.append(dict(name=f'new_native_control_g{group}_r{repeat}', measure=True,
                command=['{python}', '-u', '-B', 'joint_native_control.py', '--output', '{job}',
                         '--group', str(group), '--round', str(repeat)]))
    for group in range(9):
        jobs.append(dict(name=f'new_single_memory_g{group}', measure=False,
            command=['{python}', '-u', '-B', 'single_instance_memory.py', '--output', '{job}', '--group', str(group)]))
    path = HERE / 'reports/ADDITIONAL_20261115_PLAN.json'
    assert not path.exists()
    path.write_text(json.dumps(dict(name='ADDITIONAL_20261115', jobs=jobs), indent=2))
    with (HERE / 'logs/additional_20261115_controller.log').open('x') as log:
        child = subprocess.Popen(['/home/prmi/anaconda3/envs/zwj/bin/python', '-u', '-B', 'run_stage.py', str(path)],
            cwd=HERE, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print('ADDITIONAL_LAUNCHED', child.pid, len(jobs), flush=True)


if __name__ == '__main__':
    main()
