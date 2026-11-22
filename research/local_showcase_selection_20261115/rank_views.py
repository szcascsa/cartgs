"""基于现有局部逐帧指标筛选四Ratio共用视角；不重训、不修改历史结果。"""
from pathlib import Path
import csv
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parent
BATCH = Path('/home/prmi/zwj/realLOD/saved_experiments/dual_view_li_best_full_20261107')
RATIOS = [.01, .05, .1, .3]


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    state = load(BATCH/'LOD_PIPELINE.json')
    assert state['phase'] == 'finished'
    reports = ROOT/'reports'
    reports.mkdir(parents=True, exist_ok=True)
    protected = {str(BATCH/'LOD_PIPELINE.json'): sha(BATCH/'LOD_PIPELINE.json')}
    all_rows, groups, ratio_winners = [], [], []
    jobs = [job for job in state['jobs'] if job['dataset'] in ['Replica', 'TUM_RGBD']]
    assert len(jobs) == 26
    for job in jobs:
        assert job['state'] == 'complete'
        per_ratio = []
        for ratio in RATIOS:
            item, = [item for item in job['evaluations'] if item['scope'] == 'view_ratio' and item['ratio'] == ratio]
            output = Path(item['result']['output'])
            assert load(output/'image_preprocessing.json') == {'protocol': 'original_rgb'}
            metric_path, budget_path = output/'per_frame_metrics.csv', output/'anchor_frustum_per_view.csv'
            protected[str(metric_path)], protected[str(budget_path)] = sha(metric_path), sha(budget_path)
            metrics = list(csv.DictReader(metric_path.open()))
            budgets = list(csv.DictReader(budget_path.open()))
            assert len(metrics) == len(budgets) == item['result']['evaluated_frames']
            values = {}
            for index, (metric, budget) in enumerate(zip(metrics, budgets)):
                assert metric['synchronized_trajectory_index'] == budget['trajectory_index']
                assert metric['rgb_index'] == budget['gt_index']
                key = (metric['synchronized_trajectory_index'], metric['rgb_index'], metric['rgb_timestamp'])
                assert key not in values
                values[key] = dict(metric, evaluation_ordinal=index, output=str(output),
                                   K=int(budget['active_gaussians']), N=int(budget['total_gaussians']),
                                   candidates=int(budget['candidate_gaussians']),
                                   budget_shortfall=int(budget['budget_shortfall']))
            per_ratio.append(values)
            key, best = max(values.items(), key=lambda pair: (float(pair[1]['psnr']), -float(pair[1]['lpips'])))
            ratio_winners.append(dict(id=job['id'], ratio=ratio, **best))
        keys = set(per_ratio[0])
        assert all(set(values) == keys for values in per_ratio)
        frame_rows = []
        for key in sorted(keys, key=lambda k: int(k[0])):
            values = [lookup[key] for lookup in per_ratio]
            assert len({value['rgb_path'] for value in values}) == 1
            row = dict(id=job['id'], dataset=job['dataset'], scene=job['scene'], mode=job['mode'],
                       evaluation_ordinal=values[0]['evaluation_ordinal'],
                       trajectory_index=int(key[0]), rgb_index=int(key[1]), rgb_timestamp=key[2],
                       trajectory_timestamp=values[0]['trajectory_timestamp'], rgb_path=values[0]['rgb_path'],
                       N=values[0]['N'], candidates=values[0]['candidates'],
                       mean_psnr=statistics.mean(float(value['psnr']) for value in values),
                       min_psnr=min(float(value['psnr']) for value in values),
                       mean_ssim=statistics.mean(float(value['ssim']) for value in values),
                       mean_lpips=statistics.mean(float(value['lpips']) for value in values))
            for ratio, value in zip(RATIOS, values):
                suffix=str(int(ratio*100))
                for name in ['psnr','ssim','lpips']:
                    row[f'{name}_{suffix}'] = float(value[name])
                row[f'K_{suffix}'] = value['K']
                row[f'shortfall_{suffix}'] = value['budget_shortfall']
            frame_rows.append(row)
        ranked = sorted(frame_rows, key=lambda row: (row['mean_psnr'], -row['mean_lpips']), reverse=True)
        groups.append(dict(id=job['id'], dataset=job['dataset'], scene=job['scene'], mode=job['mode'],
                           frames=len(ranked), trained_model=job['trained_model'],
                           best=ranked[0], top=ranked[:20]))
        all_rows.extend(ranked)
    cells = {}
    for dataset in ['Replica', 'TUM_RGBD']:
        for mode in ['rgbd','mono']:
            selected=sorted([group for group in groups if group['dataset']==dataset and group['mode']==mode],
                            key=lambda group: group['best']['mean_psnr'], reverse=True)
            cells[f'{dataset}/{mode}']=selected
    assert all(sha(Path(name)) == digest for name,digest in protected.items())
    summary=dict(ratios=RATIOS,criterion='同一视角四Ratio平均PSNR降序，LPIPS次级排序；每数据集每模式各选一组',
                 groups=groups,categories=cells,frame_count=len(all_rows),
                 inputs_unchanged=True,input_sha256=protected)
    (reports/'RANKING.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    save_csv(reports/'ALL_FRAME_RANKING.csv',all_rows)
    save_csv(reports/'SCENE_BEST_SHARED_VIEW.csv',[group['best'] for group in groups])
    save_csv(reports/'PER_RATIO_BEST_VIEW.csv',ratio_winners)
    for cell, ranked in cells.items():
        print('CATEGORY',cell)
        for group in ranked:
            row=group['best']
            print(json.dumps(dict(id=group['id'],frame=row['rgb_index'],ordinal=row['evaluation_ordinal'],
                                  mean_psnr=row['mean_psnr'],mean_lpips=row['mean_lpips'],
                                  psnr=[row[f'psnr_{n}'] for n in [1,5,10,30]],
                                  K=[row[f'K_{n}'] for n in [1,5,10,30]],rgb_path=row['rgb_path']),ensure_ascii=False))
    print('RANKING_COMPLETE',len(groups),'groups',len(all_rows),'frames')


if __name__ == '__main__':
    main()
