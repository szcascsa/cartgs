"""32组冻结输入及有效全轨迹；只读，不依赖历史两组Student抽样目录。"""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT.parent
DISTILL = ['compensator_distillation_replica_all_20261108', 'compensator_distillation_vector_tum_20261112']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def views_for(scene):
    checkpoint, dataset = Path(scene['checkpoint']), Path(scene['dataset'])
    calibration = {key: json.loads((checkpoint/'cameras.json').read_text())[0][key]
                   for key in ['width', 'height', 'fx', 'fy']}
    if scene['dataset_kind'] == 'Replica':
        images = sorted((dataset/'results').glob('frame*.jpg'))
        timestamps = np.asarray([float(path.stem.removeprefix('frame')) for path in images])
    else:
        records = [line.split('#', 1)[0].split() for line in (dataset/'rgb.txt').read_text().splitlines()]
        records = [row for row in records if row]
        assert all(len(row) == 2 for row in records)
        images = [dataset/row[1] for row in records]
        timestamps = np.asarray([float(row[0]) for row in records])
    assert len(images) and np.isfinite(timestamps).all()
    trajectory = np.atleast_2d(np.loadtxt(scene['trajectory']))
    assert trajectory.shape[1] == 8 and np.isfinite(trajectory).all()
    views, seen, duplicates, unmatched = [], {}, 0, 0
    for index, record in enumerate(trajectory):
        match = int(np.abs(timestamps-record[0]).argmin())
        if abs(timestamps[match]-record[0]) > .001:
            unmatched += 1
            continue
        if match in seen:
            assert np.array_equal(record, trajectory[seen[match]]), (scene['id'], match, '不同位姿关联到同一RGB')
            duplicates += 1
            continue
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(record[4:8]).as_matrix()
        pose[:3, 3] = record[1:4]
        assert images[match].is_file(), images[match]
        views.append(dict(name=str(images[match].relative_to(dataset)), timestamp=float(record[0]),
            pose=pose.tolist(), image=str(images[match]), trajectory_index=index, rgb_index=match))
        seen[match] = index
    assert len(views) >= 200 and unmatched == 0, (scene['id'], len(views), unmatched)
    return calibration, views, dict(recorded_trajectory_rows=len(trajectory), evaluated_views=len(views),
        duplicate_identical_pose_rows=duplicates, unmatched_pose_rows=unmatched,
        protocol='all saved valid poses, native RGB; exact duplicate pose/RGB rows removed; conflicting associations rejected')


def prepare():
    import cv2
    import torch
    scenes = []
    for name in DISTILL:
        parent = EXPERIMENTS/name
        items = json.loads((parent/'reports/SCENES.json').read_text())
        report = parent/'reports/RESULTS_VALIDATION.json'
        prior_checks = {row['scene']: row for row in json.loads(report.read_text()).get('checks', [])} if report.exists() else {}
        for item in items:
            scene = dict(item)
            scene['slug'] = scene['id'].replace('/', '__')
            student = parent/'models'/scene['slug']/'continuous_calibration/best.pt'
            assert student.is_file(), student
            payload = torch.load(student, map_location='cpu')
            scene['student'] = str(student)
            scene['student_sha256'] = sha(student)
            # load_student在GPU入口还会再次核验真实Teacher指纹和结构。
            scene['student_payload_keys'] = sorted(payload)
            scene['historical_distillation_check'] = prior_checks.get(scene['id'])
            calibration, views, coverage = views_for(scene)
            for frame in [0, len(views)-1]:
                image = cv2.imread(views[frame]['image'], cv2.IMREAD_COLOR)
                assert image is not None and image.shape[:2] == (calibration['height'], calibration['width']), scene['id']
            scene.update(calibration=calibration, coverage=coverage)
            files = [*Path(scene['checkpoint']).rglob('*'), student, Path(scene['trajectory']), Path(scene['camera_config'])]
            if scene['dataset_kind'] != 'Replica':
                files.append(Path(scene['dataset'])/'rgb.txt')
            scene['input_sha256'] = {str(path): sha(path) for path in files if path.is_file()}
            scenes.append(scene)
    assert len(scenes) == len({row['id'] for row in scenes}) == 32
    assert {kind: sum(row['dataset_kind'] == kind for row in scenes) for kind in ['Replica', 'TUM_RGBD', 'VECtor']} == dict(Replica=16, TUM_RGBD=10, VECtor=6)
    # 先验收四种相机入口及TUM重复轨迹，再顺序完成其余组。
    lead = ['Replica/office2/rgbd', 'TUM_RGBD/freiburg1_xyz/rgbd', 'TUM_RGBD/freiburg1_room/mono',
            'VECtor/corner-slow/mono', 'VECtor/robot-normal/stereo', 'Replica/office0/mono']
    scenes.sort(key=lambda row: (lead.index(row['id']) if row['id'] in lead else len(lead), row['id']))
    (ROOT/'SCENES_FROZEN.json').write_text(json.dumps(scenes, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    compact = [dict(scene=row['id'], N=row['gaussians'], width=row['calibration']['width'],
        height=row['calibration']['height'], **row['coverage']) for row in scenes]
    with (ROOT/'INPUT_COVERAGE.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, list(compact[0]))
        writer.writeheader()
        writer.writerows(compact)
    print(json.dumps(dict(passed=True, groups=len(scenes), trajectory_views=sum(row['coverage']['evaluated_views'] for row in scenes),
        duplicate_rows=sum(row['coverage']['duplicate_identical_pose_rows'] for row in scenes)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    prepare()
