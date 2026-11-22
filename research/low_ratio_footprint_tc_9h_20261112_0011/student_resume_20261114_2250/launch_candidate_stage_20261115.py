"""前序验收结束后，串行构建和开发集筛选四个独立候选。"""
from pathlib import Path
import json
import subprocess

HERE = Path(__file__).resolve().parent


def main():
    state = json.loads((HERE / 'reports/ACCEPTANCE_CONTINUED_CONTROLLER.json').read_text())
    assert state['state'] == 'complete', state['state']
    jobs = [dict(name='new_thin_summary', measure=False,
                 command=['{python}', '-u', '-B', 'summarize_thin_dispatch.py'])]
    for label in ['transmittance', 'tail_cutoff', 'joint_cooperative', 'radix_grid']:
        jobs.append(dict(name=f'new_{label}_source', measure=False,
            command=['{python}', '-u', '-B', f'implement_{label}.py']))
        jobs.append(dict(name=f'new_{label}_build', measure=False,
            command=['{python}', '-u', '-B', f'{label}_runtime.py']))
        for repeat in [1, 2]:
            command = ['{python}', '-u', '-B', f'{label}_development.py', '--output', '{job}', '--round', str(repeat)]
            if label in ['transmittance', 'tail_cutoff']:
                command += ['--group', '0']
            jobs.append(dict(name=f'new_{label}_r{repeat}', measure=True, command=command))
    path = HERE / 'reports/CANDIDATES_20261115_PLAN.json'
    assert not path.exists()
    path.write_text(json.dumps(dict(name='CANDIDATES_20261115', jobs=jobs), indent=2))
    with (HERE / 'logs/candidates_20261115_controller.log').open('x') as log:
        child = subprocess.Popen(['/home/prmi/anaconda3/envs/zwj/bin/python', '-u', '-B', 'run_stage.py', str(path)],
            cwd=HERE, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print('CANDIDATES_LAUNCHED', child.pid, len(jobs), flush=True)


if __name__ == '__main__':
    main()
