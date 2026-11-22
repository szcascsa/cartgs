"""汇总跨场景蒸馏的实际质量、预算和部署检查。"""
from pathlib import Path
import csv
import hashlib
import json
import statistics as st


HERE = Path(__file__).resolve().parent
ROOT = HERE/'results' if (HERE/'results').exists() else HERE.parents[2]
REPORTS = ROOT/'reports'
STAGE_NAMES = dict(image_calibration='原图像校准',geometry_calibration='几何校准',continuous_calibration='连续Ratio校准')


def read(path):
    return json.loads(Path(path).read_text())


def main():
    scenes = read(REPORTS/'SCENES.json')
    status = read(REPORTS/'BATCH_STATUS.json')
    summary_rows = []
    run_rows = []
    for scene in scenes:
        slug = scene['slug']
        folder = REPORTS/slug
        if (folder/'QUALITY_SUMMARY.json').exists():
            quality = read(folder/'QUALITY_SUMMARY.json')
            deployment = read(folder/'DEPLOYMENT_VALIDATION.json') if (folder/'DEPLOYMENT_VALIDATION.json').exists() else {}
            preflight = read(folder/'PREFLIGHT.json')
            for record in quality['summaries']:
                row = dict(record)
                row.update(dataset=scene['dataset_kind'],mode=scene['mode'],N=scene['gaussians'],
                    total_frames=preflight['full_trajectory_frames'],anchors=quality['anchor_count'],
                    validation_frames=preflight['validation_views'],test_frames=preflight['test_views'],
                    parameters=3534,full_psnr=quality['full_psnr'],
                    deployment_passed=deployment.get('passed',False),teacher_fingerprint=preflight['fingerprint'])
                if record['student']!='teacher':
                    identity = quality['identities'][record['student']]
                    row.update(checkpoint=identity['path'],checkpoint_sha256=identity['sha256'],best_step=identity['step'])
                summary_rows.append(row)
        job = next((j for j in status['scenes'] if j['id']==scene['id']),{})
        for task in job.get('tasks',[]):
            output = ROOT/'models'/slug/task['name']
            if not (output/'training.csv').exists():
                continue
            log = list(csv.DictReader((output/'training.csv').open()))
            resources = read(output/'resources.json') if (output/'resources.json').exists() else {}
            configuration = read(output/'configuration.json')
            item = dict(scene=scene['id'],stage=task['name'],steps=len(log),
                image_updates=sum(bool(row['image_ratio']) for row in log),
                attribute_points=sum(int(row['K']) for row in log),
                image_points='image_ratio and image_local in training.csv; full_K records attribute schedule',
                local_updates=sum(row['local']=='True' for row in log),
                seconds=task.get('ended',task['started'])-task['started'],
                peak_allocated_MiB=resources.get('peak_allocated_bytes',0)/2**20 if resources else None,
                peak_reserved_MiB=resources.get('peak_reserved_bytes',0)/2**20 if resources else None,
                returncode=task.get('returncode'),configuration=json.dumps(configuration,ensure_ascii=False),
                output=str(Path('/home/prmi/zwj/experiments/compensator_distillation_cross_scene_20261108/models')/slug/task['name']))
            run_rows.append(item)
    for name,records in [('CROSS_SCENE_METRICS.csv',summary_rows),('CROSS_SCENE_TRAINING.csv',run_rows)]:
        if not records:
            continue
        fields = list(dict.fromkeys(key for row in records for key in row))
        with (REPORTS/name).open('w',newline='',encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f,fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    metadata = dict(completed=status.get('completed',0),failed=status.get('failed',0),
        conditions=len(summary_rows),training_runs=len(run_rows),
        total_updates=sum(r['steps'] for r in run_rows),images_saved=0,performance_measured=False)
    (REPORTS/'CROSS_SCENE_SUMMARY.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))


if __name__ == '__main__':
    main()
