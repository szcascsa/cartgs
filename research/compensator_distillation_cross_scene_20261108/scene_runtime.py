"""独立研究入口，使用冻结教师和完整保存的相机轨迹。"""
from common import *
from scipy.spatial.transform import Rotation
import budget_viewer
import distill_compensation


SCENES = json.loads((ROOT/'reports/SCENES.json').read_text())
BY_CHECKPOINT = {Path(s['checkpoint']).resolve():s for s in SCENES}


def full_trajectory_views(checkpoint, dataset):
    checkpoint, dataset = Path(checkpoint).resolve(), Path(dataset).resolve()
    scene = BY_CHECKPOINT[checkpoint]
    cameras = json.loads((checkpoint/'cameras.json').read_text())
    calibration = {k:cameras[0][k] for k in ['width', 'height', 'fx', 'fy']}
    if scene['dataset_kind'] == 'Replica':
        images = sorted((dataset/'results').glob('frame*.jpg'))
        timestamps = np.array([float(p.stem.removeprefix('frame')) for p in images])
    else:
        entries = [line.split('#', 1)[0].split() for line in (dataset/'rgb.txt').read_text().splitlines()]
        entries = [row for row in entries if row]
        assert all(len(row) == 2 for row in entries)
        timestamps = np.array([float(row[0]) for row in entries])
        images = [dataset/row[1] for row in entries]
    assert len(images) and np.isfinite(timestamps).all()
    records = np.atleast_2d(np.loadtxt(scene['trajectory']))
    assert records.shape[1] == 8 and np.isfinite(records).all()
    views = []
    seen = set()
    for trajectory_index, record in enumerate(records):
        image_index = int(np.abs(timestamps-record[0]).argmin())
        if abs(timestamps[image_index]-record[0]) > .001:
            continue
        if image_index in seen:
            raise ValueError('Duplicate trajectory/RGB association')
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(record[4:8]).as_matrix()
        pose[:3, 3] = record[1:4]
        assert images[image_index].is_file(), images[image_index]
        views.append(dict(name=str(images[image_index].relative_to(dataset)), timestamp=float(record[0]),
            pose=pose.tolist(), image=str(images[image_index]), trajectory_index=trajectory_index,
            rgb_index=image_index))
        seen.add(image_index)
    if len(views) < 200:
        raise RuntimeError('Insufficient full-trajectory views')
    assert len(views) == len(records), (scene['id'], len(views), len(records))
    return calibration, views


budget_viewer.load_views = full_trajectory_views
TeacherSamples = distill_compensation.TeacherSamples


def load_scene(slug):
    scene = next(s for s in SCENES if s['slug'] == slug)
    return scene, TeacherSamples(Path(scene['checkpoint']), Path(scene['dataset']))


if __name__ == '__main__':
    started = time.perf_counter()
    gpu_processes = os.popen('nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader').read()
    distill_compensation.main()
    if 'train' in sys.argv and '--output' in sys.argv:
        output = Path(sys.argv[sys.argv.index('--output')+1])
        save(output/'resources.json',dict(seconds=time.perf_counter()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gpu_processes_at_start=gpu_processes,performance_comparison=False))
