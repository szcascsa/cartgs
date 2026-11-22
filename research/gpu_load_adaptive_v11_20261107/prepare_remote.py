"""从已验收v11实验目录建立本轮完全独立的运行树。"""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[3]
V11 = Path('/home/prmi/zwj/experiments/deployment_radix_tiling_20261107')
FROZEN = Path('/home/prmi/zwj/experiments/deployment_device_counts_20261107')
TARGET = ROOT


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    if (TARGET / 'STATUS.json').exists():
        raise RuntimeError('Research run already exists; preserve its inputs')
    (TARGET / 'source/python').mkdir(parents=True, exist_ok=True)
    (TARGET / 'source/evaluation').mkdir(parents=True, exist_ok=True)
    (TARGET / 'build/mycartgs_deployment_fp32_v11').mkdir(parents=True, exist_ok=True)
    (TARGET / 'build/mycartgs_frozen_inference_v3').mkdir(parents=True, exist_ok=True)
    shutil.copytree(V11 / 'source/python', TARGET / 'source/python', dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    old = Path('/home/prmi/zwj/experiments/original_cartgs_eval_fr1_full8900_20261103_01/source/python')
    shutil.copy2(old / 'renderer.py', TARGET / 'source/original_renderer.py')
    shutil.copytree(old / 'diff-gaussian-rasterization/diff_gaussian_rasterization', TARGET / 'source/original_raster', dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    current = Path('/home/prmi/zwj/experiments/global_view_joint_lod_20261105/evaluation/python')
    shutil.copytree(current / 'diff-gaussian-rasterization/diff_gaussian_rasterization', TARGET / 'source/evaluation/diff-gaussian-rasterization/diff_gaussian_rasterization', dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    v11_src = V11 / 'build/mycartgs_deployment_fp32_v11/mycartgs_deployment_fp32_v11.so'
    frozen_candidates = list((FROZEN / 'build').glob('mycartgs_frozen_inference_v3/*.so'))
    if len(frozen_candidates) != 1: raise RuntimeError(f'Expected one frozen v3 binary, found {frozen_candidates}')
    shutil.copy2(v11_src, TARGET / 'build/mycartgs_deployment_fp32_v11/mycartgs_deployment_fp32_v11.so')
    shutil.copy2(frozen_candidates[0], TARGET / 'build/mycartgs_frozen_inference_v3/mycartgs_frozen_inference_v3.so')
    source_hashes = {str(path.relative_to(TARGET / 'source')): sha(path) for path in (TARGET / 'source').rglob('*') if path.is_file()}
    binary_hashes = {
        'mycartgs_deployment_fp32_v11.so': sha(TARGET / 'build/mycartgs_deployment_fp32_v11/mycartgs_deployment_fp32_v11.so'),
        'mycartgs_frozen_inference_v3.so': sha(TARGET / 'build/mycartgs_frozen_inference_v3/mycartgs_frozen_inference_v3.so'),
    }
    assert binary_hashes['mycartgs_deployment_fp32_v11.so'] == 'cb41244904755c25c02a1dd601b84514c83b6982c6e862043a4eb9eca7e59899'
    provenance = json.loads((V11 / 'reports/BINARY_IDENTITY.json').read_text())
    assert all(sha(TARGET / 'source' / name) == value for name, value in provenance['source_sha256'].items())
    (TARGET / 'reports').mkdir(exist_ok=True)
    (TARGET / 'reports/SOURCE_BINARY_IDENTITY.json').write_text(json.dumps({
        'base_report': str(V11 / 'DEPLOYMENT_RADIX_TILING_OPTIMIZATION_REPORT.md'),
        'v11_binary_source': str(v11_src),
        'source_sha256': source_hashes,
        'binary_sha256': binary_hashes,
    }, indent=2, ensure_ascii=False))
    previous = Path('/home/prmi/zwj/experiments/paper_intro_budget_20261107/reports/PRODUCTION_BEFORE.json')
    production = json.loads(previous.read_text())
    checkpoint = Path('/home/prmi/zwj/experiments/view_conditioned_li_20261106/runs/full_both_li/23560_shutdown/ply')
    vanilla = Path('/home/prmi/zwj/realLOD/saved_experiments/vanilla_all_datasets_20261031_2011/jobs/tum/rgbd/fr1_xyz/repeat1/seed0/attempt_2/model/10400_shutdown/ply')
    files = [Path(name) for name in production]
    for folder in (checkpoint, vanilla):
        files.extend(path for path in folder.rglob('*') if path.is_file())
        files.append(folder.parent.parent / 'CameraTrajectory_TUM.txt')
    files.extend(path for path in (TARGET / 'source').rglob('*') if path.is_file() and '__pycache__' not in path.parts)
    files.extend(path for path in (TARGET / 'build').rglob('*') if path.is_file())
    (TARGET / 'reports/PROTECTED_BEFORE.json').write_text(json.dumps({str(path): sha(path) for path in files}, indent=2))
    (TARGET / 'reports/PRODUCTION_BEFORE.json').write_text(json.dumps({name: sha(name) for name in production}, indent=2))
    print('PREPARE_REMOTE_PASS', json.dumps(binary_hashes))


if __name__ == '__main__': main()
