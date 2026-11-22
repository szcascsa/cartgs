"""冻结交付身份与小范围清理；仅删除本次失败候选的可重建对象和已导出SQLite。"""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import shutil
import time

HERE = Path(__file__).resolve().parent
R = HERE / 'reports'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    final = R / 'FINAL_DELIVERY_AUDIT.json'
    assert not final.exists()
    for name in ['RELEASE_JOINT_20261115_MERGED.json', 'RELEASE_UNSEEN_20261115_MERGED.json']:
        assert json.loads((R / name).read_text())['state'] == 'complete'
    frozen = json.loads((R / 'FROZEN_RELEASE_EXECUTION_20261115.json').read_text())
    assert all(sha(HERE / name) == value for name, value in frozen['files'].items())
    candidates = []
    for folder in ['build_async_control', 'build_cub_depth', 'build_encoded_half2', 'build_exp2_encoded',
                   'build_grouped_view', 'build_direct_pose', 'build_half_prefix', 'build_pair_coarse',
                   'build_radix_grid', 'build_selector_anchor', 'build_tile_clear', 'build_transmittance',
                   'build_wide_graph', 'build_weight_cutoff']:
        root = HERE / folder
        if root.exists():
            assert root.resolve().parent == HERE and not root.is_symlink()
            candidates.extend(root.glob('*.o'))
    for path in R.glob('release*.sqlite'):
        analysis = R / (path.stem + '_RELEASE_ANALYSIS')
        assert path.with_suffix('.nsys-rep').exists() and (analysis / 'SUMMARY.csv').exists()
        candidates.append(path)
    records = []
    for path in candidates:
        resolved = path.resolve()
        assert resolved.is_relative_to(HERE) and path.is_file() and not path.is_symlink()
        assert path.stat().st_uid == os.getuid()
        records.append(dict(path=str(resolved), bytes=path.stat().st_size, sha256=sha(path),
            reason='可由保留源码重新编译的失败候选对象' if path.suffix == '.o' else '由保留Nsight rep可重建，分析CSV已完整导出'))
    with (R / 'FINAL_CLEANUP_ALLOWLIST.json').open('x') as stream:
        json.dump(dict(time=time.time(), files=records, total_bytes=sum(x['bytes'] for x in records)), stream, indent=2, ensure_ascii=False)
    for record in records:
        Path(record['path']).unlink()
    identities = json.loads((R / 'release_execution_identity_a1/IDENTITIES.json').read_text())
    inputs = json.loads((HERE.parent / 'reports/INPUT_SHA256.json').read_text())
    identity_rows = []
    for item in identities:
        entry = dict(scene=item['scene'], checkpoint=item['checkpoint'],
            checkpoint_sha256=inputs.get(item['checkpoint']), student=item['student'], student_sha256=item['student_sha256'],
            map_fingerprint=item['fingerprint'], N=item['N'], student_parameters=item['student_parameter_count'],
            selector_parameters=item['selector_parameter_count'], network_structure=item['structure'],
            selector_backend=item['selector_backend'], student_backend=item['compensator_backend'])
        if entry['checkpoint_sha256'] is None:
            entry['checkpoint_sha256'] = {name: value for name, value in inputs.items()
                if Path(name).is_relative_to(Path(item['checkpoint']))}
        identity_rows.append(entry)
    (R / 'FINAL_STUDENT_MANIFEST.json').write_text(json.dumps(identity_rows, indent=2, ensure_ascii=False))
    binaries = {}
    for folder in ['build_joint_release', 'build_joint_final', 'build_encoded_sign', 'build_tail_strong']:
        binaries.update({str(path): dict(sha256=sha(path), bytes=path.stat().st_size) for path in (HERE / folder).glob('*.so')})
    images = [str(path) for path in HERE.rglob('*') if path.is_file() and path.suffix.lower() in
        ['.png', '.jpg', '.jpeg', '.bmp', '.webp'] and 'figures' not in path.relative_to(HERE).parts]
    assert not images
    result = dict(passed=True, time=time.time(), local_time=datetime.now().astimezone().isoformat(),
        frozen_execution_files=len(frozen['files']), frozen_execution_unchanged=True,
        student_identities=len(identity_rows), binaries=binaries, render_images=images,
        cleanup_files=len(records), reclaimed_bytes=sum(x['bytes'] for x in records), free_bytes=shutil.disk_usage(HERE).free,
        rollback='source_joint_final/build_joint_final与冻结F2原目录全部保留',
        no_commit_push_deploy=True, no_training=True)
    final.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print('FINAL_DELIVERY_AUDIT_PASS', len(records), result['reclaimed_bytes'], result['free_bytes'])


if __name__ == '__main__':
    main()
