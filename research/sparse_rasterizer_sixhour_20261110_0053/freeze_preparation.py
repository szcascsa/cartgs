"""新增相机固定计算候选在保留场景开放前锁定独立身份。"""
from pathlib import Path
import hashlib,json,time

HERE=Path(__file__).resolve().parent
path=HERE/'reports/FROZEN_PREPARATION.json'
assert not path.exists()
assert json.loads((HERE/'reports/camera_study01/SUMMARY.json').read_text())['passed']
job=json.loads((HERE/'reports/jobs/camera_study01.json').read_text())
assert job['measurement_valid'] and job['foreign_active_samples']==0
value=dict(frozen=True,time=time.time(),sources={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    for name in ['camera_preparation.py','final_preparation_benchmark.py']},
    source_renderer_identity='reports/FROZEN_CANDIDATE.json revision2; renderer binary unchanged',
    tune_scenes_only=True,heldout_scenes=19,variant='F = G + fixed intrinsics preparation',
    invalidation=['width','height','FoVx','FoVy','znear','zfar','device','stream','default_dtype'],
    bounded_storage='one FP32 4x4 projection tensor per preparer; no frame, pose, visibility or feature cache',
    threshold='相同Full逐场景两轮中位不慢于A/B/G；逐位相机矩阵/选点/RGB/radii；不隐藏空集')
path.write_text(json.dumps(value,indent=2,ensure_ascii=False));print('PREPARATION_FROZEN',value['time'])
