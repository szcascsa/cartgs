"""打包数值证据和模型位置；不复制模型，也不保存渲染图片。"""
from pathlib import Path
import hashlib
import json
import tarfile

R = Path(__file__).resolve().parent
B = R.parent/'dual_view_li_best_full_20261107'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    state = json.loads((R/'LOD_PIPELINE.json').read_text())
    original = json.loads((B/'LOD_PIPELINE.json').read_text())
    models = []
    cases = []
    for label, jobs in [('li01', [j for j in original['jobs'] if j['id'] in [x['id'] for x in state['jobs']]]),
                        ('li05', state['jobs'])]:
        for job in jobs:
            if 'trained_model' not in job:
                continue
            model = Path(job['trained_model'])
            ply, = model.glob('*_shutdown/ply')
            files = [ply/'selector.pt', ply/'real_lod_state.json', *(ply/'point_cloud').glob('iteration_*/point_cloud.ply')]
            models.append(dict(scene=job['scene'], li_weight=label, model=str(model),
                               checkpoint=str(ply), files={str(f): sha(f) for f in files},
                               original_full=job['full_job']['result']['checkpoint']))
            for item in job['evaluations']:
                if item['state'] == 'complete':
                    cases.append((label, job['scene'], item))
    (R/'reports/MODELS.json').write_text(json.dumps(models, indent=2)+'\n')
    with tarfile.open(R/'LI05_RESULTS_EVIDENCE.tar.gz', 'w:gz') as tar:
        for folder in ['reports', 'configs']:
            for file in (R/folder).rglob('*'):
                if file.is_file():
                    tar.add(file, arcname=str(file.relative_to(R)))
        for file in R.glob('*.py'):
            tar.add(file, arcname='helpers/'+file.name)
        tar.add(R/'LOD_PIPELINE.json', arcname='EXTRA_CONTROLS_STATE.json')
        for label, scene, item in cases:
            folder = Path(item['result']['output'])
            for name in ['per_frame_metrics.csv', 'anchor_frustum_per_view.csv', 'metric_summary.json',
                         'image_preprocessing.json', 'exit.json']:
                file = folder/name
                if file.is_file():
                    tar.add(file, arcname=f"per_view/{label}/{scene}/{item['scope']}_{item['ratio']}/{name}")
    print('PACKAGED', len(cases), 'cases', (R/'LI05_RESULTS_EVIDENCE.tar.gz').stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
