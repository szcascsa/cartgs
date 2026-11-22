"""核验并收集全场景CaRtGS未校正原图口径数值结果。"""
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parent
SAVED = ROOT.parent
VANILLA = SAVED/'vanilla_all_datasets_20261031_2011'
VECTOR = SAVED/'dual_view_li_vector_best_full_20261107'
BATCHES = [SAVED/'dual_view_li_best_full_20261107', VECTOR]
METRICS = ['psnr', 'ssim', 'lpips']


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def migrated(path):
    return Path(path.replace('/home/prmi/zwj/experiments/vanilla_all_datasets_20261031_2011', str(VANILLA)))


def canonical(scene):
    dataset, mode, name = scene.split('/')
    dataset = {'tum': 'TUM_RGBD', 'replica': 'Replica', 'vector': 'VECtor'}.get(dataset, dataset)
    if dataset == 'TUM_RGBD':
        name = name.replace('fr1_', 'freiburg1_').replace('fr2_', 'freiburg2_').replace('fr3_', 'freiburg3_')
    return f'{dataset}/{name}/{mode}'


def main():
    assert read(ROOT/'REEVALUATION_COMPLETE.json')['complete']
    jobs = []
    protected = {}
    for batch in BATCHES:
        state = read(batch/'LOD_PIPELINE.json')
        assert state['phase'] == 'finished' and all(j['state'] == 'complete' for j in state['jobs'])
        protected[str(batch/'LOD_PIPELINE.json')] = sha(batch/'LOD_PIPELINE.json')
        protected[str(batch/'reports/LOD_METRICS.csv')] = sha(batch/'reports/LOD_METRICS.csv')
        for job in state['jobs']:
            jobs.append(dict(job, batch=str(batch)))
    assert len(jobs) == 32
    job_by_id = {j['id']: j for j in jobs}
    protected[str(VANILLA/'reports/PER_RUN.csv')] = sha(VANILLA/'reports/PER_RUN.csv')
    candidates = defaultdict(list)
    for row in csv.DictReader((VANILLA/'reports/PER_RUN.csv').open()):
        identity = canonical(row['scene'])
        if identity not in job_by_id or row['state'] != 'complete':
            continue
        assert row['seed'] == '0' and row['phase'] == 'FULL'
        value = dict(id=row['id'], group=identity, repeat=int(row['repeat']),
                     checkpoint=str(migrated(row['checkpoint'])), evaluation=str(migrated(row['evaluation'])),
                     gaussians=int(row['N']), frames=int(row['views']),
                     evaluation_protocol='historical_full_trajectory',
                     **{name: float(row[name]) for name in METRICS})
        assert Path(value['checkpoint']).is_dir()
        for name in METRICS:
            file = Path(value['evaluation'])/(name+'.txt')
            values = [float(line) for line in file.read_text().splitlines() if line.strip()]
            assert len(values) == value['frames']
            assert abs(sum(values)/len(values)-value[name]) < 1e-6
            protected[str(file)] = sha(file)
        candidates[identity].append(value)
    assert set(candidates) == set(job_by_id)
    write(ROOT/'data/SOURCE_HASHES_BEFORE.json', protected)
    selected = {}
    for identity, values in sorted(candidates.items()):
        selected[identity] = max(values, key=lambda value: (value['psnr'], -value['repeat']))
        selected[identity]['completed_repetitions'] = len(values)
    write(ROOT/'data/VANILLA_CANDIDATES.json', dict(candidates))
    groups = []
    verified = 0
    for job in sorted(jobs, key=lambda j: j['id']):
        records = []
        assert len(job['evaluations']) == 23
        for item in job['evaluations']:
            assert item['state'] == 'complete'
            summary = item['result']
            output = Path(summary['output'])
            actual = read(output/'metric_summary.json')
            assert read(output/'image_preprocessing.json') == {'protocol': 'original_rgb'}
            for name in METRICS:
                assert math.isfinite(actual[name]) and actual[name] == summary[name]
            with (output/'per_frame_metrics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) == summary['evaluated_frames']
            for name in METRICS:
                assert abs(sum(float(row[name]) for row in rows)/len(rows)-summary[name]) < 1e-9
            gaussian_count = int(summary['gaussian_count'])
            records.append(dict(scope=item['scope'], ratio=item['ratio'], ratio_kind=item['kind'],
                                gaussian_count=gaussian_count, actual_K_mean=summary['actual_K_mean'],
                                actual_K_min=summary['actual_K_min'], actual_K_max=summary['actual_K_max'],
                                frames=summary['evaluated_frames'], output=str(output),
                                evaluation_protocol='cartgs_original_rgb_full_trajectory',
                                **{name: summary[name] for name in METRICS}))
            verified += 1
        baseline = selected[job['id']]
        groups.append(dict(id=job['id'], dataset=job['dataset'], scene=job['scene'], mode=job['mode'],
                           full_source_winner=job['winner'], fine_attempt=Path(job['attempt']).name,
                           fine_steps=job['target_steps'], trained_model=job['trained_model'],
                           source_full=job['full_evaluation']['result'], vanilla=baseline, ours=records,
                           source_full_selection_protocol=job.get('full_psnr_protocol', 'historical_selection_original_rgb'),
                           frames_match=baseline['frames'] == records[0]['frames']))
    assert verified == 736
    after = {name: sha(Path(name)) for name in protected}
    assert after == protected
    write(ROOT/'data/SOURCE_HASHES_AFTER.json', after)
    bundle = dict(generated_hk=datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(), groups=groups,
                  protocols=dict(selection='每组按Full全轨迹PSNR最高重复取值，其余指标绑定该次',
                                 gaussian_count='Full为地图总点数N，LoD为实际送入渲染的K，局部为跨视角均值',
                                 stereo='双方均使用CaRtGS未校正原图，关闭尺度对齐；不改变训练和选模结果',
                                 coverage='各自完整估计轨迹，标注帧数，不进行共同帧重聚合'),
                  verification=dict(ours_metric_records=verified, groups=len(groups),
                                    original_rgb_reevaluations=69, source_hashes=len(protected),
                                    source_hashes_unchanged=True,
                                    datasets=dict(Counter(g['dataset'] for g in groups))))
    write(ROOT/'data/ALL_SCENE_RESULTS.json', bundle)
    write(ROOT/'COLLECTION_COMPLETE.json', bundle['verification'])
    print('COLLECTION_COMPLETE', json.dumps(bundle['verification']), flush=True)


if __name__ == '__main__':
    main()
