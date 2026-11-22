"""验证整理完整性与待启动状态，不执行训练。"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

R = Path('/home/prmi/zwj/realLOD/ablation')
B = Path('/home/prmi/zwj/realLOD/baseline')
V = Path('/home/prmi/zwj/realLOD/vanilla')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def own_files(root):
    return {str(p.relative_to(root)): sha(p) for p in root.rglob('*')
            if p.is_file() and not p.is_symlink()
            and not any(a.is_symlink() for a in p.parents if a != root)
            and '__pycache__' not in p.parts}


def main():
    assert (R/'HOLD').exists() and not (R/'START_AUTHORIZED.json').exists()
    state = read(R/'LOD_PIPELINE.json')
    assert state['phase'] == 'prepared_held' and all(j['state'] == 'pending' for j in state['jobs'])
    assert not state['protocol']['master_updates'] and 'excluded_dataset' not in state['protocol']
    for folder in ['runs', 'evaluation', 'validation']:
        assert not list((R/folder).iterdir()), folder
    own_commands = {str(R/name) for name in ['run_batch.py', 'supervisor.py', 'pipeline.py',
                                            'runtime/bin/lod_from_full']}
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            command = (proc/'cmdline').read_bytes().decode().split('\x00')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        assert not own_commands.intersection(command), (proc.name, command)
    with (R/'logs/cpu_tests.log').open('w') as stream:
        result = subprocess.run([sys.executable, '-m', 'unittest', 'test_queue', 'test_ablation', '-v'],
                                cwd=R, stdout=stream, stderr=subprocess.STDOUT)
    assert result.returncode == 0
    write(R/'reports/CPU_TESTS.json', dict(passed=True, count=16, training_started=False))
    before = read(R/'PROTECTED_BEFORE.json')
    after = {k: {n: sha(Path(n)) for n in values} for k, values in before.items()}
    assert before == after, '原Full、生产或baseline模型发生变化'
    write(R/'reports/PROTECTED_PREPARATION_AFTER.json', after)
    moved = read(B/'RESULTS_RELOCATION.json')
    count = 0
    for batch in moved['protected'].values():
        for name, expected in batch.items():
            path = Path(name)
            assert sha(path) == expected['sha256'] and path.stat().st_ino == expected['inode']
            assert path.stat().st_size == expected['size']
            count += 1
    assert count == 2928
    current = own_files(R/'source')
    reference = own_files(B/'dual_view_li_best_full_20261107/source')
    changed = sorted(n for n in set(current)|set(reference) if current.get(n) != reference.get(n))
    expected = sorted(read(R/'reports/SOURCE_CHANGES.json'))
    assert changed == expected, changed
    write(R/'reports/SOURCE_HASHES.json', current)
    write(R/'reports/QUEUE_SOURCE_HASHES.json', {p.name: sha(p) for p in R.glob('*.py')})
    assert read(R/'reports/BOOTSTRAP.json')['phase'] == 'build_complete'
    assert read(R/'reports/LINKAGE_VALIDATION.json')['passed']
    for name, expected_hash in read(R/'reports/BUILD_HASHES.json').items():
        assert sha(R/name) == expected_hash
    receipt = dict(passed=True, checked_unix=time.time(), groups=32, pending=32,
                   fine_steps=637377, planned_trajectory_evaluations=736,
                   cpu_tests_passed=16, independently_compiled=True,
                   changed_source_files=changed, protected_files_unchanged=sum(map(len, after.values())),
                   relocation_hash_inode_verified=count, training_started=False,
                   gpu_tests_run=False, scene_smoke_started=False, held=True,
                   requires_explicit_user_start=True)
    write(R/'reports/PREPARED.json', receipt)
    lines = ['# 结果整理与Master固定消融准备报告', '',
             'baseline已完整迁移32组双网络真实视角＋LI=0.1结果，包含736项全轨迹评价。',
             'vanilla已完整迁移101次成功运行及1次历史跳过，保留Replica、TUM、VECtor和3DLite。',
             '迁移使用同盘rename；旧saved_experiments路径保留符号链接。2928文件SHA256、大小和inode复验通过。', '',
             '消融包含Replica16组、TUM10组、VECtor6组，637377步Fine、736项评价。',
             '每组绑定baseline使用的最佳原Full checkpoint、相机、seed0和步数，不从Fine后的模型继续。',
             '唯一算法差异：Full／全局LoD／局部LoD三路全部detach Master；只更新Selector、Compensator和Anchor Feature。',
             'Full渲染、原Loss及反向仍执行，Full继续提供LI教师；参数更新仍是三路之后一次共享Adam。',
             '训练ratio=[0.01,0.05,0.1,0.3,0.5,0.7]；未见ratio=[0.03,0.08,0.2,0.4,0.6]只评价。',
             '保留原候选池、预算、LI=0.1、健康停止、OOM重试上限及双目左图校正；不保存渲染图片。', '',
             '仅修改独立source内3个C++文件，头文件110项逐位一致，未改生产或baseline源码。',
             '独立增量编译3个对象、Mapper动态库和Fine入口成功；其余对象只读复用。',
             'ldd/readelf确认入口加载ablation/runtime/lib/libgaussian_mapper.so，依赖全部找到。',
             '16项CPU测试通过，涵盖单一配置差异、32组Full起点与步数匹配、重试语义和启动锁。',
             '所有6类Master参数设置requires_grad=false；训练结束必须逐位不变、无梯度、无Adam状态。',
             '运行时还必须验证网络更新及Anchor Feature Adam状态存在。', '',
             '**当前没有启动训练、GPU单元测试或场景smoke。** runs/evaluation/validation均为空；HOLD存在，启动授权不存在。',
             '待用户明确说跑后，监督器先执行GPU验证，再按各数据集/相机入口进行6步smoke和初态逐位校验，通过才正式训练。',
             '尚未执行的GPU和场景验证不构成已通过项；失败时队列停止并保留现场。', '',
             '启动入口（本次未执行）：`/home/prmi/anaconda3/envs/zwj/bin/python /home/prmi/zwj/realLOD/ablation/start_queue.py --start`。',
             '主队列：`LOD_PIPELINE.json`；协议：`reports/ABLATION_PROTOCOL.md`；验收：`reports/PREPARED.json`。']
    (R/'reports/PREPARATION_REPORT.md').write_text('\n\n'.join(lines)+'\n')
    (B/'README.md').write_text('# 双网络真实视角＋LI baseline\n\n'
        '已完成32组：Replica16、TUM10、VECtor6。共736项全轨迹评价。\n\n'
        'Replica、TUM_RGBD、VECtor为按数据集浏览的链接；完整批次、配置、日志和失败历史均保留。\n\n'
        'Replica/TUM指标：`dual_view_li_best_full_20261107/reports/LOD_METRICS.csv`。\n\n'
        'VECtor指标：`dual_view_li_vector_best_full_20261107/reports/LOD_METRICS.csv`。\n\n'
        '旧saved_experiments路径继续可用；迁移核验见RESULTS_RELOCATION.json。\n')
    (V/'README.md').write_text('# 纯Vanilla CaRtGS结果\n\n'
        '完整批次为vanilla_all_datasets_20261031_2011，101次成功，1次历史跳过。\n\n'
        'Replica、TUM_RGBD、VECtor、3DLite链接分别指向完整批次jobs内对应数据集。\n\n'
        '保留原模型、日志、指标、配置；旧saved_experiments路径继续可用。\n')
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
