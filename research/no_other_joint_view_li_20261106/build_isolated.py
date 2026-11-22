"""只在独立研究目录编译；复用只读第三方依赖和未修改的 CUDA 库。"""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = Path('/home/prmi/zwj/experiments/fr1xyz_mixed_ratio_pair_20261104')
PROJECT = Path('/home/prmi/zwj/mycartgs')


def execute(name, command):
    report_path = ROOT/'reports'/f'{name}.json'
    if '-c' in command and report_path.exists():
        old = json.loads(report_path.read_text())
        output = Path(command[command.index('-o')+1])
        newest = max(p.stat().st_mtime for folder in ['include', 'improvements', 'cuda_rasterizer']
                     for p in (ROOT/'source'/folder).rglob('*.h'))
        newest = max(newest, Path(command[-1]).stat().st_mtime)
        if old['returncode'] == 0 and old['command'] == command and output.exists() and output.stat().st_mtime >= newest:
            print(name, 'unchanged', flush=True)
            return
    started = time.monotonic()
    with (ROOT/'logs'/f'{name}.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    report = {'command': command, 'returncode': result.returncode, 'seconds': time.monotonic()-started}
    report_path.write_text(json.dumps(report, indent=2))
    print(name, result.returncode, round(report['seconds'], 2), flush=True)
    if result.returncode:
        raise RuntimeError(f'{name}: see {ROOT}/logs/{name}.log')


def compile_command(template, source, output):
    command = [arg.replace(str(TEMPLATE), str(ROOT)) for arg in template]
    for flag in ['-MT', '-MF', '-o', '-c']:
        at = command.index(flag)
        del command[at:at+2]
    command = [arg for arg in command if arg != '-MD']
    return command + ['-o', str(output), '-c', str(source)]


def main():
    assert ROOT == Path('/home/prmi/zwj/experiments/no_other_joint_view_li_20261106'), ROOT
    for name in ['objects', 'logs', 'reports', 'runtime/lib', 'runtime/bin']:
        (ROOT/name).mkdir(parents=True, exist_ok=True)
    for name in ['ORB-SLAM3', 'tinyply', 'simple-knn']:
        dest = ROOT/'source/third_party'/name
        dest.parent.mkdir(exist_ok=True)
        if not dest.exists():
            dest.symlink_to(PROJECT/'third_party'/name, target_is_directory=True)
    for source in (TEMPLATE/'runtime/lib').iterdir():
        if source.is_file() and source.name != 'libgaussian_mapper.so':
            shutil.copy2(source, ROOT/'runtime/lib'/source.name)
    flags = json.loads((TEMPLATE/'compile_gaussian_mapper.json').read_text())['command']
    mapper = json.loads((TEMPLATE/'link_mapper.json').read_text())['command']
    objects = [arg for arg in mapper if arg.endswith('.o')]
    replacements, tasks = {}, []
    for obj in objects:
        name = Path(obj).name
        source = ROOT/'source'/('third_party/tinyply/tinyply.cpp' if name == 'tinyply.cpp.o' else 'src/'+name[:-2] + ('' if name.endswith('.cpp.o') else '.cpp'))
        output = ROOT/'objects'/name
        replacements[obj] = str(output)
        tasks.append((name, compile_command(flags, source, output)))
    aux = ROOT/'objects/view_budget_training.cpp.o'
    tasks.append(('view_budget_training.cpp.o', compile_command(flags, ROOT/'source/src/view_budget_training.cpp', aux)))
    # All Mapper C++ translation units use this snapshot; no stale class layout.
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(execute, 'compile_'+name, command) for name, command in tasks]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    old_cuda=Path('/home/prmi/zwj/experiments/real_lod_20261103')
    cuda_archive=ROOT/'runtime/lib/libtransform_field_cuda.a'
    shutil.copyfile('/home/prmi/zwj/experiments/view_conditioned_joint_20261106/runtime/lib/libtransform_field_cuda.a',cuda_archive)
    command = [replacements.get(arg, arg).replace(str(TEMPLATE), str(ROOT)) for arg in mapper]
    command = [str(cuda_archive) if arg==str(old_cuda/'build/libtransform_field_cuda.a') else arg for arg in command]
    command.insert(command.index('-o'), str(aux))
    execute('link_mapper', command)
    entry_flags = json.loads((TEMPLATE/'compile_entry.json').read_text())['command']
    entry_link = json.loads((TEMPLATE/'link_entry.json').read_text())['command']
    targets = {
        'residual48_no_other_test': 'tests/anchor_selector/residual48_no_other_test.cpp',
        'view_budget_training_test': 'tests/anchor_selector/view_budget_training_test.cpp',
        'real_lod_test': 'tests/anchor_selector/real_lod_test.cpp',
        'lod_health_guard_test': 'tests/lod_health_guard_test.cpp',
        'lod_from_full': 'research/global_view_joint_lod_20261105/lod_from_full.cpp',
        'local_importance_test': 'tests/anchor_selector/local_importance_test.cpp',
        'view_budget_residual_test': 'tests/anchor_selector/view_budget_residual_test.cpp',
        'view_residual_test': 'tests/anchor_selector/view_residual_test.cpp',
        'view_input_test': 'tests/anchor_selector/view_input_test.cpp',
        'joint_view_conditioning_test': 'tests/anchor_selector/joint_view_conditioning_test.cpp',
        'joint_view_cuda_parity_test': 'tests/anchor_selector/joint_view_cuda_parity_test.cpp',
    }
    for name, relative in targets.items():
        obj = ROOT/'objects'/f'{name}.o'
        command = compile_command(entry_flags, ROOT/'source'/relative, obj)
        command.insert(1, f'-DCARTGS_TEST_SOURCE_DIR="{ROOT}/source"')
        execute('compile_'+name, command)
        command = [arg.replace(str(TEMPLATE), str(ROOT)) for arg in entry_link]
        command[command.index('-o')+1] = str(ROOT/'runtime/bin'/name)
        command = [str(obj) if arg == str(ROOT/'objects/lod_pair.o') else arg for arg in command]
        execute('link_'+name, command)
    manifest = {}
    for folder in ['objects', 'runtime']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file():
                manifest[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    (ROOT/'reports/BUILD_HASHES.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
