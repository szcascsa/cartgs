"""串行构建下一批独立候选；开发集结果决定是否保留，不并发扰动正式测速。"""
from pathlib import Path
import json
import subprocess

HERE = Path(__file__).resolve().parent


def main():
    state = json.loads((HERE / 'reports/TAIL_VALIDATION_20261115_CONTROLLER.json').read_text())
    assert state['state'] == 'complete', state['state']
    jobs = []
    for label in ['tail_strong', 'encoded_half2', 'cub_depth', 'tile_clear', 'half_prefix',
                  'joint_incremental', 'full_pose']:
        jobs.append(dict(name=f'new_{label}_source', measure=False,
            command=['{python}', '-u', '-B', f'implement_{label}.py']))
        jobs.append(dict(name=f'new_{label}_build', measure=False,
            command=['{python}', '-u', '-B', f'{label}_runtime.py']))
        if label == 'cub_depth':
            jobs.append(dict(name='new_cub_depth_edges', measure=False,
                command=['{python}', '-u', '-B', 'cub_depth_edges.py', '--output', '{job}', '--modes', '182']))
        for repeat in [1, 2]:
            command = ['{python}', '-u', '-B', f'{label}_development.py', '--output', '{job}', '--round', str(repeat)]
            if label in ['tail_strong', 'encoded_half2', 'cub_depth']:
                command += ['--group', '0']
            jobs.append(dict(name=f'new_{label}_r{repeat}', measure=True, command=command))
    path = HERE / 'reports/DEEP_CANDIDATES_20261115_PLAN.json'
    assert not path.exists()
    path.write_text(json.dumps(dict(name='DEEP_CANDIDATES_20261115', jobs=jobs), indent=2))
    with (HERE / 'logs/deep_candidates_20261115_controller.log').open('x') as log:
        child = subprocess.Popen(['/home/prmi/anaconda3/envs/zwj/bin/python', '-u', '-B', 'run_stage.py', str(path)],
            cwd=HERE, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print('DEEP_CANDIDATES_LAUNCHED', child.pid, len(jobs), flush=True)


if __name__ == '__main__':
    main()
