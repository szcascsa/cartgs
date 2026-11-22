"""独立增量编译固定Master消融，不启动任何训练或GPU测试。"""
from pathlib import Path
import difflib
import hashlib
import json
import subprocess
import time

R = Path('/home/prmi/zwj/realLOD/ablation')
B = Path('/home/prmi/zwj/realLOD/saved_experiments/dual_view_li_best_full_20261107')
CHANGED = ['src/gaussian_mapper.cpp', 'src/real_lod.cpp',
           'research/global_view_joint_lod_20261105/lod_from_full.cpp']


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(name, command):
    output = Path(command[command.index('-o')+1])
    assert output.is_relative_to(R) and not output.is_symlink()
    started = time.monotonic()
    with (R/'logs'/f'{name}.log').open('w') as stream:
        result = subprocess.run(command, cwd=R, stdout=stream, stderr=subprocess.STDOUT)
    write(R/'reports'/f'{name}.json', dict(command=command, returncode=result.returncode,
                                        seconds=time.monotonic()-started))
    print(name, result.returncode, round(time.monotonic()-started, 2), flush=True)
    assert result.returncode == 0, name


def main():
    assert (R/'HOLD').exists() and not (R/'START_AUTHORIZED.json').exists()
    # 删除被外层RealLoD enabled条件恒定排除的旧限制，修正对应诊断。
    path = R/'source/src/gaussian_mapper.cpp'
    text = path.read_text().replace(
        'real_lod_full_loss_weight_ < 0.f ||\n'
        '      (real_lod_full_loss_weight_ > 0.f && selector_detach_gaussian_parameters_ && !real_lod_.enabled)',
        'real_lod_full_loss_weight_ < 0.f')
    text = text.replace('RealLoD.full_loss_weight must be nonnegative; Full supervision requires a live Master',
                        'RealLoD.full_loss_weight must be finite and nonnegative')
    path.write_text(text)
    path = R/'source/src/real_lod.cpp'
    text = path.read_text().replace('Full Fine supervision requires a live Master',
                                   'Full Fine supervision requires an active Fine phase and positive weight')
    path.write_text(text)
    # 复用未改动对象的前提：所有头文件保持逐位一致。
    headers = {}
    for path in (B/'source').rglob('*'):
        if path.suffix in ['.h', '.hpp', '.cuh'] and path.is_file() and not any(
                p.is_symlink() for p in path.parents if p != B):
            relative = path.relative_to(B/'source')
            assert sha(path) == sha(R/'source'/relative), str(relative)
            headers[str(relative)] = sha(path)
    write(R/'reports/UNCHANGED_HEADERS.json', headers)
    diff = []
    changes = {}
    for relative in CHANGED:
        original = B/'source'/relative
        current = R/'source'/relative
        changes[relative] = dict(before_sha256=sha(original), after_sha256=sha(current))
        diff.extend(difflib.unified_diff(original.read_text().splitlines(True),
                    current.read_text().splitlines(True), fromfile='baseline/'+relative,
                    tofile='ablation/'+relative))
    (R/'reports/MASTER_DETACH.diff').write_text(''.join(diff))
    write(R/'reports/SOURCE_CHANGES.json', changes)
    for name in ['gaussian_mapper.o', 'real_lod.o', 'lod_from_full']:
        command = read(B/'reports'/f'compile_{name}.json')['command']
        command = [arg.replace(str(B), str(R)) for arg in command]
        execute('compile_'+name, command)
    for name in ['mapper', 'lod_from_full']:
        command = read(B/'reports'/f'link_{name}.json')['command']
        new = []
        for arg in command:
            if arg.endswith('.o'):
                if Path(arg).name in ['gaussian_mapper.o', 'real_lod.o', 'lod_from_full.o']:
                    arg = str(R/'objects'/Path(arg).name)
                else:
                    assert Path(arg).is_file(), arg
            else:
                arg = arg.replace(str(B), str(R))
            if arg.endswith('/libtransform_field_cuda.a'):
                arg = str(R/'runtime/lib/libtransform_field_cuda.a')
            new.append(arg)
        execute('link_'+name, new)
    reused = {}
    for arg in read(R/'reports/link_mapper.json')['command']:
        if arg.endswith(('.o', '.a')) and not Path(arg).is_relative_to(R):
            reused[arg] = sha(Path(arg))
    write(R/'reports/READ_ONLY_BUILD_INPUTS.json', reused)
    write(R/'reports/BUILD_HASHES.json', {str(p.relative_to(R)): sha(p)
          for folder in ['runtime', 'objects'] for p in (R/folder).rglob('*') if p.is_file()})
    write(R/'reports/BOOTSTRAP.json', dict(phase='build_complete', training_started=False,
          scene_smoke_started=False, gpu_tests_run=False, incremental_build=True,
          unchanged_header_count=len(headers), readonly_baseline_objects=len(reused)))
    print('BUILD_COMPLETE_TRAINING_HELD', flush=True)


if __name__ == '__main__':
    main()
