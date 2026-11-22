"""按冻结门限统一选择渲染器，联合Student仅使用同一冻结权重并保留所有负例。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import subprocess
import time

HERE = Path(__file__).resolve().parent


def main():
    root = HERE / 'reports'
    assert json.loads((root / 'RELEASE_RENDER_20261115_CONTROLLER.json').read_text())['state'] == 'complete'
    destination = root / 'SELECTED_RASTER_20261115.json'
    assert not destination.exists()
    subprocess.run(['/home/prmi/anaconda3/envs/zwj/bin/python', '-B', 'summarize_strong_validation.py'], cwd=HERE, check=True)
    results = []
    for prefix, variants in [('TAIL', [('alpha2', 203)]), ('STRONG', [('alpha3', 301), ('alpha4', 302)])]:
        stats = pd.read_csv(root / (prefix + '_SCENE_RATIO.csv'))
        quality = pd.read_csv(root / (prefix + '_QUALITY_CONDITIONS.csv'))
        for variant, mode in variants:
            q = quality[quality.variant == variant]
            mean = q.groupby('workload').psnr_loss.mean()
            perf = stats[(stats.variant == variant) & stats.condition.isin([f'local{x:g}' for x in [.1,.2,.3,.4,.5]])]
            by_round = perf.groupby(['workload', 'round']).previous182_speedup.agg(lambda x: np.exp(np.log(x).mean()))
            gain = float(np.exp(np.log(perf.previous182_speedup).mean()))
            majority = perf.groupby(['workload', 'round']).previous182_speedup.agg(lambda x: (x > 1).mean())
            passed = bool(mean.max() <= .2 and q.psnr_loss.max() <= .5 and by_round.min() > 1 and majority.min() > .5)
            results.append(dict(variant=variant, mode=mode, quality_means=mean.to_dict(),
                worst_condition=float(q.psnr_loss.max()), worst_frame=float(q.worst_frame_loss.max()),
                previous182_speedup=gain, worst_round_speedup=float(by_round.min()), passed=passed))
    accepted = sorted([row for row in results if row['passed']], key=lambda row: row['previous182_speedup'], reverse=True)
    chosen = accepted[0] if accepted else dict(mode=182, variant='previous182', passed=True, previous182_speedup=1.)
    destination.write_text(json.dumps(dict(time=time.time(), **chosen, candidates=results,
        decision='全场景统一配置；依据已冻结质量和双轮多数收益规则选择，不修改任何渲染参数。'), indent=2, ensure_ascii=False))
    jobs = [dict(name='release_execution_identity', measure=False,
                 command=['{python}', '-u', '-B', 'release_identity.py', '--output', '{job}']),
            dict(name='release_joint_edges', measure=False,
                 command=['{python}', '-u', '-B', 'release_joint_edges.py', '--output', '{job}'])]
    for group in range(9):
        jobs.append(dict(name=f'release_joint_quality_g{group}', measure=False,
            command=['{python}', '-u', '-B', 'release_joint.py', '--output', '{job}', '--group', str(group), '--quality']))
    for repeat in [1, 2]:
        for group in (range(9) if repeat == 1 else reversed(range(9))):
            jobs.append(dict(name=f'release_joint_g{group}_r{repeat}', measure=True,
                command=['{python}', '-u', '-B', 'release_joint.py', '--output', '{job}', '--group', str(group), '--round', str(repeat)]))
    for repeat in [1, 2]:
        for workload, groups in [('student_lod', 9), ('nested', 16)]:
            for group in (range(groups) if repeat == 1 else reversed(range(groups))):
                jobs.append(dict(name=f'release_thin_{workload}_g{group}_r{repeat}', measure=True,
                    command=['{python}', '-u', '-B', 'release_thin_control.py', '--output', '{job}',
                        '--group', str(group), '--round', str(repeat), '--workload', workload]))
    for tool in ['memcheck', 'initcheck', 'synccheck']:
        jobs.append(dict(name='release_joint_' + tool, measure=False,
            command=['/usr/local/cuda-11.8/bin/compute-sanitizer', '--tool', tool, '--error-exitcode', '99',
                     '--print-limit', '10', '{python}', '-u', '-B', 'release_joint_edges.py', '--output', '{job}', '--sanitizer']))
    for repeat in [1, 2]:
        for group in (range(9) if repeat == 1 else reversed(range(9))):
            jobs.append(dict(name=f'release_prepare_g{group}_r{repeat}', measure=True,
                command=['{python}', '-u', '-B', 'release_preparation.py', '--output', '{job}', '--group', str(group), '--round', str(repeat)]))
    for group in range(9):
        jobs.append(dict(name=f'release_memory_g{group}', measure=False,
            command=['{python}', '-u', '-B', 'release_memory.py', '--output', '{job}', '--group', str(group)]))
    for group in [0, 1]:
        jobs.append(dict(name=f'release_profile_g{group}', measure=False,
            command=['/usr/local/cuda-11.8/bin/nsys', 'profile', '--trace=cuda,nvtx', '--cuda-graph-trace=node',
                     '--capture-range=cudaProfilerApi', '--capture-range-end=stop', '--sample=none',
                     '--force-overwrite=false', '--output', '{root}/reports/{job}', '{python}', '-u', '-B',
                     'release_profile.py', '--group', str(group)]))
    jobs.append(dict(name='release_raster_profile', measure=False,
        command=['/usr/local/cuda-11.8/bin/nsys', 'profile', '--trace=cuda,nvtx', '--cuda-graph-trace=node',
                 '--capture-range=cudaProfilerApi', '--capture-range-end=stop', '--sample=none',
                 '--force-overwrite=false', '--output', '{root}/reports/{job}', '{python}', '-u', '-B',
                 'release_raster_profile.py']))
    for repeat in [1, 2]:
        jobs.append(dict(name=f'release_stress_r{repeat}', measure=True,
            command=['{python}', '-u', '-B', 'stress_raster_final.py', '--output', '{job}', '--engine',
                     'strong' if chosen['mode'] in [301,302] else 'tail', '--mode', str(chosen['mode']), '--round', str(repeat)]))
    jobs.append(dict(name='release_integrity', measure=False,
        command=['{python}', '-u', '-B', 'verify_integrity.py', '--output', '{job}.json']))
    path = root / 'RELEASE_JOINT_20261115_PLAN.json'
    assert not path.exists()
    path.write_text(json.dumps(dict(name='RELEASE_JOINT_20261115', jobs=jobs), indent=2))
    with (HERE / 'logs/release_joint_20261115_controller.log').open('x') as log:
        child = subprocess.Popen(['/home/prmi/anaconda3/envs/zwj/bin/python', '-u', '-B', 'run_stage.py', str(path)],
            cwd=HERE, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print('RELEASE_JOINT_LAUNCHED', chosen, child.pid, len(jobs), flush=True)


if __name__ == '__main__':
    main()
