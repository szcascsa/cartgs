"""锁定冻结配置在新视角的独立复验；等原控制器完成，不覆盖任务。"""
from pathlib import Path
import json
import subprocess

HERE = Path(__file__).resolve().parent


def main():
    r = HERE / 'reports'
    assert json.loads((r / 'RELEASE_JOINT_20261115_MERGED.json').read_text())['state'] == 'complete'
    jobs = [dict(name='release_profile_export', measure=False,
        command=['{python}', '-u', '-B', 'export_release_profiles.py', '--output', '{job}'])]
    for group in range(9):
        jobs.append(dict(name=f'unseen_quality_g{group}', measure=False,
            command=['{python}', '-u', '-B', 'release_unseen_views.py', 'joint', '--output', '{job}', '--group', str(group), '--quality']))
    for group in range(16):
        jobs.append(dict(name=f'unseen_nested_quality_g{group}', measure=False,
            command=['{python}', '-u', '-B', 'release_unseen_nested_quality.py', '--output', '{job}', '--group', str(group)]))
    for repeat in [1, 2]:
        for group in (range(9) if repeat == 1 else reversed(range(9))):
            jobs.append(dict(name=f'unseen_joint_g{group}_r{repeat}', measure=True,
                command=['{python}', '-u', '-B', 'release_unseen_views.py', 'joint', '--output', '{job}',
                    '--group', str(group), '--round', str(repeat)]))
        for workload, groups in [('student_lod', 9), ('nested', 16)]:
            for group in (range(groups) if repeat == 1 else reversed(range(groups))):
                jobs.append(dict(name=f'unseen_thin_{workload}_g{group}_r{repeat}', measure=True,
                    command=['{python}', '-u', '-B', 'release_unseen_views.py', 'thin', '--output', '{job}',
                        '--group', str(group), '--round', str(repeat), '--workload', workload]))
    plan = r / 'RELEASE_UNSEEN_20261115_PLAN.json'
    with plan.open('x') as stream:
        json.dump(dict(name='RELEASE_UNSEEN_20261115', jobs=jobs, final_validation=True,
            views=dict(continuous=[.58, '.58n+1'], jump=[.23, .81]),
            no_more_parameter_tuning=True), stream, indent=2)
    with (HERE / 'logs/release_unseen_controller.log').open('x') as log:
        child = subprocess.Popen(['/home/prmi/anaconda3/envs/zwj/bin/python', '-u', '-B', 'run_stage.py', str(plan)],
            cwd=HERE, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print('UNSEEN_VALIDATION_LAUNCHED', child.pid, len(jobs), flush=True)


if __name__ == '__main__':
    main()
