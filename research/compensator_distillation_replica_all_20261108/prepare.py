"""准备Replica八场景、单目和RGB-D的统一蒸馏，核验复用结果。"""
from pathlib import Path
import hashlib
import json
import shutil


ROOT = Path(__file__).resolve().parents[3]
OLD = Path('/home/prmi/zwj/experiments/compensator_distillation_cross_scene_20261108')
SAVED = Path('/home/prmi/zwj/realLOD/saved_experiments/dual_view_li_best_full_20261107')
NAMES = ['office0','office1','office2','office3','office4','room0','room1','room2']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not (ROOT/'reports/SCENES.json').exists()
    assert shutil.disk_usage(ROOT).free > 5*2**30
    protocol = json.loads((OLD/'reports/PROTOCOL.json').read_text())
    old_scenes = {s['id']:s for s in json.loads((OLD/'reports/SCENES.json').read_text())}
    queue = json.loads((SAVED/'LOD_PIPELINE.json').read_text())
    expected = {f'Replica/{name}/{mode}' for name in NAMES for mode in ['mono','rgbd']}
    jobs = {j['id']:j for j in queue['jobs'] if j['dataset']=='Replica'}
    assert set(jobs) == expected
    protected = {}
    reused_files = {}
    scenes = []
    for name in NAMES:
        for mode in ['mono','rgbd']:
            key = f'Replica/{name}/{mode}'
            job = jobs[key]
            assert job['state']=='complete',key
            model = Path(job['trained_model'])
            checkpoint, = model.glob('*_shutdown/ply')
            cloud, = (checkpoint/'point_cloud').glob('iteration_*/point_cloud.ply')
            with cloud.open('rb') as stream:
                while True:
                    line = stream.readline().decode().strip()
                    if line.startswith('element vertex '):
                        count = int(line.split()[-1])
                    if line=='end_header':
                        break
            scene = dict(id=key,dataset_kind='Replica',scene=name,mode=mode,checkpoint=str(checkpoint),
                dataset=job['full_evaluation']['data'],camera_config=job['camera_config'],
                trajectory=str(model/'CameraTrajectory_TUM.txt'),gaussians=count,prior_full_psnr=job['full_psnr'],
                slug=key.replace('/','__'),reused=False)
            for path in [*checkpoint.rglob('*'),model/'CameraTrajectory_TUM.txt',Path(job['camera_config'])]:
                if path.is_file():
                    protected[str(path)] = dict(bytes=path.stat().st_size,sha256=sha(path))
            if key in old_scenes:
                prior = old_scenes[key]
                assert all(prior[k]==scene[k] for k in ['checkpoint','dataset','trajectory','gaussians'])
                source_folder = OLD/'reports'/scene['slug']
                quality = json.loads((source_folder/'QUALITY_SUMMARY.json').read_text())
                assert json.loads((source_folder/'DEPLOYMENT_VALIDATION.json').read_text())['passed']
                for stage in protocol['stages']:
                    source = OLD/'models'/scene['slug']/stage['name']
                    config = json.loads((source/'configuration.json').read_text())
                    for key2,expected_value in [('seed',0),('learning_rate',stage['lr']),('image_every',stage['image_every']),
                            ('image_weight',stage['image_weight']),('residual_weight',stage['residual_weight']),
                            ('ratio_jitter',stage['ratio_jitter']),('ratios',protocol['train_ratios'])]:
                        assert config[key2]==expected_value,(key,key2)
                    assert len((source/'training.csv').read_text().splitlines())==stage['steps']+1
                    if stage['name'] in quality['identities']:
                        assert sha(source/'best.pt')==quality['identities'][stage['name']]['sha256']
                shutil.copytree(source_folder,ROOT/'reports'/scene['slug'])
                shutil.copytree(OLD/'models'/scene['slug'],ROOT/'models'/scene['slug'])
                for folder in [source_folder,OLD/'models'/scene['slug']]:
                    for path in folder.rglob('*'):
                        if path.is_file():
                            reused_files[str(path)] = dict(bytes=path.stat().st_size,sha256=sha(path))
                scene.update(reused=True,reused_from=str(OLD),reuse_check='teacher checkpoint, stage configs, steps, student SHA, evaluation and CUDA validation')
            scenes.append(scene)
            print(scene['id'],count,'REUSED' if scene['reused'] else 'NEW',flush=True)
    protocol.update(scenes=[s['id'] for s in scenes],scope='Replica eight scenes, mono and RGB-D',
        reused_scenes=[s['id'] for s in scenes if s['reused']],new_scenes=[s['id'] for s in scenes if not s['reused']],
        planned_new_updates=sum(not s['reused'] for s in scenes)*protocol['total_updates_per_scene'],
        source_protocol=str(OLD/'reports/PROTOCOL.json'),user_mode_interpretation='Replica follows previously confirmed mono plus RGB-D')
    for prefix in ['PRODUCTION','INPUTS','LEGACY_INPUTS','LARGE_INPUTS']:
        before = json.loads((ROOT/'reports'/f'{prefix}_BEFORE.json').read_text())
        for path,value in before.items():
            assert sha(path)==(value['sha256'] if isinstance(value,dict) else value),path
    (ROOT/'reports/SCENES.json').write_text(json.dumps(scenes,indent=2,ensure_ascii=False))
    (ROOT/'reports/PROTOCOL.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False))
    (ROOT/'reports/SCENE_INPUTS_BEFORE.json').write_text(json.dumps(protected,indent=2))
    shutil.copy2(OLD/'reports/BATCH_STATUS.json',ROOT/'reports/REUSE_SOURCE_BATCH_STATUS.json')
    (ROOT/'reports/REUSED_INPUTS_BEFORE.json').write_text(json.dumps(reused_files,indent=2))
    source = {str(p.relative_to(ROOT/'source')):sha(p) for p in (ROOT/'source').rglob('*')
        if p.is_file() and '__pycache__' not in str(p)}
    (ROOT/'reports/SOURCE_BEFORE.json').write_text(json.dumps(source,indent=2))
    print('REPLICA_PREPARED',len(scenes),'new',len(protocol['new_scenes']),'reused',len(protocol['reused_scenes']),flush=True)


if __name__=='__main__':
    main()
