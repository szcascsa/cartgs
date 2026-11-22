"""锁定四个新场景的教师、完整轨迹与保护清单。"""
from pathlib import Path
import hashlib
import json
import math
import shutil


ROOT = Path(__file__).resolve().parents[3]
SAVED = Path('/home/prmi/zwj/realLOD/saved_experiments')
IDS = ['TUM_RGBD/freiburg1_desk/rgbd', 'Replica/office0/rgbd',
    'Replica/office3/rgbd', 'VECtor/corner-slow/mono']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not (ROOT/'reports/SCENES.json').exists()
    assert shutil.disk_usage(ROOT).free > 5*2**30
    for prefix in ['PRODUCTION', 'INPUTS', 'LEGACY_INPUTS', 'LARGE_INPUTS']:
        path = ROOT/'reports'/f'{prefix}_BEFORE.json'
        before = json.loads(path.read_text())
        for name, value in before.items():
            expected = value['sha256'] if isinstance(value, dict) else value
            assert sha(name) == expected, name
    jobs = {}
    for folder in ['dual_view_li_best_full_20261107', 'dual_view_li_vector_best_full_20261107']:
        queue = json.loads((SAVED/folder/'LOD_PIPELINE.json').read_text())
        for job in queue['jobs']:
            if job['id'] in IDS:
                jobs[job['id']] = job
    assert set(jobs) == set(IDS)
    scenes = []
    protected = {}
    for name in IDS:
        job = jobs[name]
        assert job['state'] == 'complete', name
        model = Path(job['trained_model'])
        checkpoint, = model.glob('*_shutdown/ply')
        cloud, = (checkpoint/'point_cloud').glob('iteration_*/point_cloud.ply')
        with cloud.open('rb') as f:
            header = []
            while True:
                line = f.readline().decode().strip()
                header.append(line)
                if line == 'end_header':
                    break
        n = int(next(v for v in header if v.startswith('element vertex ')).split()[-1])
        scene = dict(id=name, dataset_kind=job['dataset'], scene=job['scene'], mode=job['mode'],
            checkpoint=str(checkpoint), dataset=job['full_evaluation']['data'],
            camera_config=job['camera_config'], trajectory=str(model/'CameraTrajectory_TUM.txt'),
            gaussians=n, prior_full_psnr=job['full_psnr'])
        scene['slug'] = name.replace('/', '__')
        scenes.append(scene)
        for p in [*checkpoint.rglob('*'), model/'CameraTrajectory_TUM.txt', Path(job['camera_config'])]:
            if p.is_file():
                protected[str(p)] = dict(bytes=p.stat().st_size, sha256=sha(p))
        print(json.dumps(scene, ensure_ascii=False), flush=True)
    ratios = [.01, .05, .1, .3, .5, .7]
    stages = [
        dict(name='warmup', steps=6000, lr=.001, image_every=20, image_weight=1.,
            residual_weight=1., validation_metric='combined', validate_every=500, ratio_jitter=0., paired=False),
        dict(name='image_calibration', steps=2000, lr=.001, image_every=5, image_weight=1000.,
            residual_weight=1., validation_metric='combined', validate_every=500, ratio_jitter=0., paired=False),
        dict(name='geometry_calibration', steps=6000, lr=.0002, image_every=1, image_weight=200.,
            residual_weight=.01, validation_metric='image_mse', validate_every=300, ratio_jitter=0., paired=False),
        dict(name='continuous_calibration', steps=3600, lr=.0001, image_every=1, image_weight=200.,
            residual_weight=.01, validation_metric='image_mse', validate_every=300, ratio_jitter=.5, paired=True)]
    protocol = dict(scenes=IDS, seed=0, train_ratios=ratios, non_anchor_ratios=[.03, .08, .2, .4, .6],
        continuous_validation_ratios=ratios+[math.sqrt(a*b) for a,b in zip(ratios,ratios[1:])],
        stages=stages, total_updates_per_scene=sum(s['steps'] for s in stages), batch_points=4096,
        calibration_cases=12, test_views=32, validation_views=3, validation_policy='every fifth frame pool, three uniformly spaced validation views, remainder test',
        gt_used_for_selection=False, per_scene_hyperparameter_tuning=False,
        architecture='32/32 and four direct heads', parameters=3534,
        quality_gate=dict(mean_psnr_loss=.01, p99_psnr_loss=.05, worst_psnr_loss=.1),
        checkpoint_policy='each stage initializes from previous stage best, new Adam and RNG',
        compare_students=['image_calibration/best.pt', 'geometry_calibration/best.pt', 'continuous_calibration/best.pt'],
        renderer_changes=False, speed_measurement=False, render_images_saved=0,
        gpu_memory_policy='one distillation process at a time, other processes preserved')
    (ROOT/'reports/SCENES.json').write_text(json.dumps(scenes, indent=2, ensure_ascii=False))
    (ROOT/'reports/PROTOCOL.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False))
    (ROOT/'reports/SCENE_INPUTS_BEFORE.json').write_text(json.dumps(protected, indent=2))
    source = {str(p.relative_to(ROOT/'source')):sha(p) for p in (ROOT/'source').rglob('*')
        if p.is_file() and '__pycache__' not in str(p)}
    (ROOT/'reports/SOURCE_BEFORE.json').write_text(json.dumps(source, indent=2))


if __name__ == '__main__':
    main()
