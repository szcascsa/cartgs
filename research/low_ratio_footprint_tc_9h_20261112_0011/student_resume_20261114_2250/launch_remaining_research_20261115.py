"""完成已有验收后，串行测试剩余两个准备候选；不创建重复等待控制器。"""
from pathlib import Path
import json
import subprocess

HERE = Path(__file__).resolve().parent


def main():
    state = json.loads((HERE / 'reports/ADDITIONAL_CONTINUED_CONTROLLER.json').read_text())
    assert state['state'] == 'complete', state['state']
    jobs = []
    for label in ['direct_pose', 'selector_anchor', 'wide_graph']:
        jobs.append(dict(name=f'new_{label}_source', measure=False,
            command=['{python}', '-u', '-B', f'implement_{label}.py']))
        jobs.append(dict(name=f'new_{label}_build', measure=False,
            command=['{python}', '-u', '-B', f'{label}_runtime.py']))
        for repeat in [1, 2]:
            jobs.append(dict(name=f'new_{label}_r{repeat}', measure=True,
                command=['{python}', '-u', '-B', f'{label}_development.py', '--output', '{job}', '--round', str(repeat)]))
    path = HERE / 'reports/REMAINING_RESEARCH_20261115_PLAN.json'
    assert not path.exists()
    path.write_text(json.dumps(dict(name='REMAINING_RESEARCH_20261115', jobs=jobs), indent=2))
    with (HERE / 'logs/remaining_research_20261115_controller.log').open('x') as log:
        child = subprocess.Popen(['/home/prmi/anaconda3/envs/zwj/bin/python', '-u', '-B', 'run_stage.py', str(path)],
            cwd=HERE, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print('REMAINING_RESEARCH_LAUNCHED', child.pid, len(jobs), flush=True)


if __name__ == '__main__':
    main()
