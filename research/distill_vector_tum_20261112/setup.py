"""在独立目录建立 TUM 和 VECtor 学生蒸馏队列。"""
from pathlib import Path
import hashlib
import json
import shutil
import time

ROOT = Path('/home/prmi/zwj/experiments/compensator_distillation_vector_tum_20261112')
REPLICA = Path('/home/prmi/zwj/experiments/compensator_distillation_replica_all_20261108')
SAVED = Path('/home/prmi/zwj/realLOD/saved_experiments')
PIPELINES = ['dual_view_li_best_full_20261107', 'dual_view_li_vector_best_full_20261107']
RERUN = 'dual_view_li_vector_stereo_rerun_20261111'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    assert not ROOT.exists(), ROOT
    assert shutil.disk_usage(ROOT.parent).free > 5*2**30
    ROOT.mkdir()
    for name in ['reports', 'logs', 'models', 'build']:
        (ROOT/name).mkdir()
    shutil.copytree(REPLICA/'source', ROOT/'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    selected = {}
    for name in PIPELINES:
        queue = json.loads((SAVED/name/'LOD_PIPELINE.json').read_text())
        for job in queue['jobs']:
            if job['dataset'] in ('TUM_RGBD','VECtor'):
                selected[job['id']] = job
    queue = json.loads((SAVED/RERUN/'LOD_PIPELINE.json').read_text())
    for job in queue['jobs']:
        if job['dataset']=='VECtor' and job['mode']=='stereo' and job['state']=='complete':
            selected[job['id']] = job
    assert len(selected)==16 and len([j for j in selected.values() if j['dataset']=='VECtor'])==6
    scenes = []
    protected = {}
    for key,job in sorted(selected.items()):
        assert job['state']=='complete', key
        model = Path(job['trained_model'])
        checkpoint, = model.glob('*_shutdown/ply')
        cloud, = (checkpoint/'point_cloud').glob('iteration_*/point_cloud.ply')
        with cloud.open('rb') as stream:
            count = None
            while True:
                line = stream.readline().decode().strip()
                if line.startswith('element vertex '):
                    count = int(line.split()[-1])
                if line=='end_header':
                    break
            assert count and count>0
        trajectory = model/'CameraTrajectory_TUM.txt'
        dataset = Path(job['full_evaluation']['data'])
        config = Path(job['camera_config'])
        assert trajectory.is_file() and config.is_file() and (dataset/'rgb.txt').is_file(), key
        for path in [*checkpoint.rglob('*'),trajectory,config]:
            if path.is_file():
                protected[str(path)] = dict(bytes=path.stat().st_size,sha256=sha(path))
        scenes.append(dict(id=key,dataset_kind=job['dataset'],scene=job['scene'],mode=job['mode'],
            checkpoint=str(checkpoint),dataset=str(dataset),camera_config=str(config),trajectory=str(trajectory),
            gaussians=count,prior_full_psnr=job['full_psnr'],slug=key.replace('/','__'),reused=False))
    protocol = json.loads((REPLICA/'reports/PROTOCOL.json').read_text())
    protocol.update(scenes=[s['id'] for s in scenes],scope='TUM RGB-D mono/rgbd; VECtor mono/stereo',
        reused_scenes=[],new_scenes=[s['id'] for s in scenes],planned_new_updates=16*17600,
        source_protocol=str(REPLICA/'reports/PROTOCOL.json'))
    (ROOT/'reports/SCENES.json').write_text(json.dumps(scenes,indent=2,ensure_ascii=False))
    (ROOT/'reports/PROTOCOL.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False))
    (ROOT/'reports/SCENE_INPUTS_BEFORE.json').write_text(json.dumps(protected,indent=2))
    for prefix in ['PRODUCTION','INPUTS','LEGACY_INPUTS','LARGE_INPUTS']:
        before = json.loads((REPLICA/'reports'/f'{prefix}_BEFORE.json').read_text())
        for path,value in before.items():
            assert sha(Path(path))==(value['sha256'] if isinstance(value,dict) else value),path
        (ROOT/'reports'/f'{prefix}_BEFORE.json').write_text(json.dumps(before,indent=2))
    (ROOT/'reports/REUSED_INPUTS_BEFORE.json').write_text('{}')
    source = {str(p.relative_to(ROOT/'source')):sha(p) for p in (ROOT/'source').rglob('*') if p.is_file()}
    (ROOT/'reports/SOURCE_BEFORE.json').write_text(json.dumps(source,indent=2))
    (ROOT/'reports/PARALLEL_SETTINGS.json').write_text(json.dumps(dict(max_workers=2)))
    state = dict(started=time.time(),status='running',scenes=[],reused=0)
    (ROOT/'reports/BATCH_STATUS.json').write_text(json.dumps(state,indent=2))
    print('PREPARED',len(scenes),flush=True)
    for scene in scenes:
        print(scene['id'],scene['gaussians'],scene['checkpoint'],flush=True)


if __name__=='__main__':
    main()
