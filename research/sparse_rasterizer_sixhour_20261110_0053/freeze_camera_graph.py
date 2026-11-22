"""Q候选在保留场景开放前冻结；仅改变相机准备，渲染二进制不变。"""
from pathlib import Path
import hashlib,json,time

HERE=Path(__file__).resolve().parent
target=HERE/'reports/FROZEN_CAMERA_GRAPH.json';assert not target.exists()
for name in ['camera_graph01','camera_graph_edges01','sanitizer_camera_graph01']:
    assert json.loads((HERE/'reports'/name/'SUMMARY.json').read_text())['passed']
job=json.loads((HERE/'reports/jobs/camera_graph01.json').read_text())
assert job['measurement_valid'] and job['foreign_active_samples']==0
value=dict(frozen=True,time=time.time(),sources={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in
    ['camera_preparation.py','camera_graph_preparation.py','final_camera_graph_benchmark.py']},
    variant='Q = G Rasterizer + 固定内参 + 相机FP32 BMM/inv_ex执行图 + 原_linalg_check_errors',
    renderer_identity='FROZEN_CANDIDATE.json revision2; regular and graph binaries unchanged',
    tuning_scenes_only=True,heldout_scenes=19,output_cloned_every_frame=True,inverse_check_every_frame=True,
    no_pose_or_visibility_cache=True,bound='每实例一个4x4执行图；内参/device/Stream改变立即失效重建',
    threshold='所有24组两轮Full中位不慢于A/B/G/F；相机/ID/RGB/radii逐位一致；保留所有反例')
target.write_text(json.dumps(value,indent=2,ensure_ascii=False));print('CAMERA_GRAPH_FROZEN',value['time'])
