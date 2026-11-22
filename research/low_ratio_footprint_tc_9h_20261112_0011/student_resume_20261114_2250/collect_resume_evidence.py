"""下载所需的小型源码和文本证据，不打包模型、图片或大型Trace。"""
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent


def main():
    files = set()
    reports = HERE / 'reports'
    for path in reports.iterdir():
        if path.is_file() and path.suffix in ['.json', '.csv', '.gz']:
            files.add(path)
    for folder in ['source_encoded_sign', 'source_joint_final', 'source_overflow_prefix']:
        for path in (HERE / folder).rglob('*'):
            if path.is_file() and path.suffix in ['.cpp', '.cu', '.cuh', '.h']:
                files.add(path)
    for name in ['student_raster_profile01_a1_ANALYSIS_V2']:
        files.update(path for path in (reports / name).rglob('*') if path.is_file())
    for path in reports.glob('*_ANALYSIS'):
        files.update(value for value in path.rglob('*') if value.is_file() and value.suffix in ['.csv', '.gz'])
    for path in reports.iterdir():
        if path.is_dir() and (path.name.startswith(('student_overflow_prefix_r', 'student_chunk_loop_r',
                                                   'student_fast_publish_r'))):
            files.update(value for value in path.iterdir() if value.is_file() and value.suffix in ['.csv', '.json'])
    destination = reports / 'RESUME_EVIDENCE_20261115_0244.tar.gz'
    assert not destination.exists()
    with tarfile.open(destination, 'w:gz') as archive:
        for path in sorted(files):
            archive.add(path, arcname=str(path.relative_to(HERE)), recursive=False)
    print('EVIDENCE_READY', len(files), destination.stat().st_size, flush=True)


if __name__ == '__main__':
    main()
