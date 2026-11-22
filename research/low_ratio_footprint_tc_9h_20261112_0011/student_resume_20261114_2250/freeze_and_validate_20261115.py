"""第六小时前冻结，不依据保留场景调参；构建联合模块并双轮复验3/4弱尾候选。"""
from pathlib import Path
import hashlib
import json
import subprocess
import time

HERE = Path(__file__).resolve().parent


def main():
    r = HERE / 'reports'
    for label in ['TAIL_VALIDATION_20261115', 'DEEP_CANDIDATES_20261115']:
        assert json.loads((r / (label + '_CONTROLLER.json')).read_text())['state'] == 'complete'
    frozen = r / 'FROZEN_FINAL_CONTENDERS_20261115.json'
    assert not frozen.exists()
    binary = next((HERE / 'build_tail_strong').glob('*.so'))
    frozen.write_text(json.dumps(dict(time=time.time(), development=['Replica/office2/rgbd', 'Replica/room0/rgbd'],
        raster_modes=[301, 302], network=dict(direct_pose=True, view_lanes=16, native_full=True),
        decision='按全测试质量门限先筛选；合格候选中仅保留双轮多数条件获益者，以主要Ratio几何平均速度选择同一全局配置，禁止逐场景调参。无合格增量则回退182。',
        quality=dict(mean=.2, scene_ratio=.5), binary=str(binary),
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        sources={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (HERE/'source_tail_strong').rglob('*') if p.is_file()},
        no_more_raster_tuning=True), ensure_ascii=False, indent=2))
    jobs = [dict(name='release_source', measure=False, command=['{python}', '-u', '-B', 'implement_joint_release.py']),
            dict(name='release_build', measure=False, command=['{python}', '-u', '-B', 'joint_release_runtime.py'])]
    jobs.append(dict(name='release_tail_edges', measure=False,
        command=['{python}', '-u', '-B', 'tail_strong_edges.py', '--output', '{job}', '--modes', '301,302']))
    for repeat in [1, 2]:
        for label, groups in [('formal', 9), ('nested', 16)]:
            for group in (range(groups) if repeat == 1 else reversed(range(groups))):
                jobs.append(dict(name=f'release_render_{label}_g{group}_r{repeat}', measure=True,
                    command=['{python}', '-u', '-B', f'tail_strong_{label}.py', '--output', '{job}',
                             '--group', str(group), '--round', str(repeat)]))
    for tool in ['memcheck', 'initcheck', 'synccheck']:
        jobs.append(dict(name='release_tail_'+tool, measure=False,
            command=['/usr/local/cuda-11.8/bin/compute-sanitizer', '--tool', tool, '--error-exitcode', '99',
                     '--print-limit', '10', '{python}', '-u', '-B', 'tail_strong_edges.py', '--output', '{job}',
                     '--modes', '301,302', '--sanitizer']))
    path = r / 'RELEASE_RENDER_20261115_PLAN.json'
    assert not path.exists()
    path.write_text(json.dumps(dict(name='RELEASE_RENDER_20261115', jobs=jobs), indent=2))
    with (HERE / 'logs/release_render_20261115_controller.log').open('x') as log:
        child = subprocess.Popen(['/home/prmi/anaconda3/envs/zwj/bin/python', '-u', '-B', 'run_stage.py', str(path)],
            cwd=HERE, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print('FINAL_CONTENDERS_FROZEN', child.pid, len(jobs), flush=True)


if __name__ == '__main__':
    main()
