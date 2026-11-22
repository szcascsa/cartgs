"""只读复用fr1_desk RGB-D baseline，导出第469帧Full作为Anchor示意底图。"""
from pathlib import Path
import hashlib
import json
import os
import runpy
import sys

root=Path(__file__).resolve().parent
batch=Path('/home/prmi/zwj/realLOD/baseline/dual_view_li_best_full_20261107')
job=next(j for j in json.loads((batch/'LOD_PIPELINE.json').read_text())['jobs']
         if j['id']=='TUM_RGBD/freiburg1_desk/rgbd')
model=Path(job['trained_model']).resolve()
shutdown,=model.glob('*_shutdown')
inputs=[model/'CameraTrajectory_TUM.txt',shutdown/'ply/cameras.json',
        *list((shutdown/'ply/point_cloud').glob('iteration_*/point_cloud.ply'))]
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
before={str(f):sha(f) for f in inputs}
frame_set=root/'FRAME_SET.json';frame_set.write_text(json.dumps(dict(gt_indices=[469])))
sys.path[:0]=[str(batch/'source/python'),str(batch/'evaluation_runtime')]
import torch
import cv2
from evo.tools.settings import SETTINGS
SETTINGS.plot_backend='Agg'
torch.manual_seed(0);torch.cuda.manual_seed_all(0)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
original_write=cv2.imwrite
def write_png(path,pixels,*args,**kwargs):
    assert original_write(str(Path(path).with_suffix('.png')),pixels,*args,**kwargs)
    return True
cv2.imwrite=write_png
script=batch/'source/python/run_metrics_only.py'
sys.argv=[str(script),str(model),job['full_evaluation']['data'],'--shutdown-model',shutdown.name,
    '--output_path',str(root/'full_0'),'--evaluation-frame-set',str(frame_set)]
result=runpy.run_path(str(script),run_name='__main__')
assert all(sha(Path(f))==digest for f,digest in before.items())
(root/'BACKGROUND_PROVENANCE.json').write_text(json.dumps(dict(dataset='TUM_RGBD',scene='freiburg1_desk',
    mode='rgbd',model=str(model),gt_index=469,psnr=float(result['psnr_list'][0]),
    gaussian_count=int(result['gaussians'].get_xyz.shape[0]),input_sha256=before,
    note='真实Full渲染底图；后续Anchor示意为非实际尺寸的概念叠加'),ensure_ascii=False,indent=2))
print('BACKGROUND_PASS',list((root/'full_0/image').glob('*.png')),flush=True)
