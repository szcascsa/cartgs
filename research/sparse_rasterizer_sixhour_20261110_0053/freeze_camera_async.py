"""R异步上传候选在保留场景开放前冻结。"""
from pathlib import Path
import hashlib,json,time

HERE=Path(__file__).resolve().parent
path=HERE/'reports/FROZEN_CAMERA_ASYNC.json';assert not path.exists()
for name in ['async_camera_edges01','async_camera_study01','sanitizer_camera_async01']:
    assert json.loads((HERE/'reports'/name/'SUMMARY.json').read_text())['passed']
job=json.loads((HERE/'reports/jobs/async_camera_study01.json').read_text());assert job['measurement_valid'] and job['foreign_active_samples']==0
result=dict(frozen=True,time=time.time(),sources={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in
    ['camera_preparation.py','camera_async_preparation.py','final_camera_async_benchmark.py']},
    variant='R = G Rasterizer + Q camera Graph with non_blocking H2D input transfer',
    renderer_identity='FROZEN_CANDIDATE.json revision2 binaries unchanged',tuning_scenes_only=True,heldout_scenes=19,
    inverse_check_every_frame=True,same_stream_copy_and_graph_dependency=True,no_delayed_checks=True,
    threshold='24场景两轮Full对A/B/G/F/Q均不慢；相机/ID/RGB/radii逐位一致；否则保留Q')
path.write_text(json.dumps(result,ensure_ascii=False,indent=2));print('ASYNC_CAMERA_FROZEN',result['time'])
