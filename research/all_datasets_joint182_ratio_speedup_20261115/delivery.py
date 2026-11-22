"""测量全部结束后打包图、摘要及复现脚本；大体积原始计时保留服务器。"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    queue = json.loads((ROOT/'QUEUE.json').read_text())
    validation = json.loads((ROOT/'reports/ANALYSIS_VALIDATION.json').read_text())
    assert queue['state'] == 'complete' and validation['passed'] and validation['repeats'] == 3
    assert validation['groups'] == 32 and validation['total_views'] == 59507
    assert validation['raw_timing_rows'] == 2142252
    for filename, expected in validation['sources'].items():
        assert sha(Path(filename)) == expected, filename
    subprocess.run([sys.executable, '-B', str(ROOT/'plot_dataset_modes.py'), '--first-pass'], check=True)
    subprocess.run([sys.executable, '-B', str(ROOT/'plot_dataset_modes.py')], check=True)
    files = [ROOT/name for name in ['inputs.py', 'worker.py', 'worker_passes.py', 'job_runner.py',
        'controller.py', 'controller_passes.py', 'analyze_passes.py', 'status.py',
        'delivery.py', 'plot_dataset_modes.py', 'SCENES_FROZEN.json', 'INPUT_COVERAGE.csv', 'QUEUE.json', 'CONTROLLER.json',
        'FIRST_PASS_READY.json', 'USER_FIRST_PASS_REQUEST.json', 'ALL_DATASETS_MODE182_RATIO_SPEEDUP_REPORT.md']]
    for root in [ROOT/'figures', ROOT/'reports', ROOT/'first_pass']:
        files.extend(path for path in root.rglob('*') if path.is_file() and path.name != 'PER_VIEW_SPEEDUP.csv')
    files.extend(ROOT/'logs'/name for name in ['controller.log', 'controller_passes.log',
        'first_pass_analysis.log', 'final_analysis.log'])
    files = sorted(set(files))
    manifest = {str(path.relative_to(ROOT)): sha(path) for path in files}
    manifest_path = ROOT/'DELIVERY_SHA256.json'
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    archive = ROOT/'ALL_DATASETS_MODE182_DELIVERY.tar.gz'
    with tarfile.open(archive, 'w:gz', compresslevel=3) as pack:
        for path in [*files, manifest_path]:
            pack.add(path, arcname=str(path.relative_to(ROOT)))
    digest = sha(archive)
    print('ARCHIVE_SHA_A', digest[:32])
    print('ARCHIVE_SHA_B', digest[32:])
    print('ARCHIVE_BYTES', archive.stat().st_size)
    print('DELIVERY_FILES', len(files))


if __name__ == '__main__':
    main()
