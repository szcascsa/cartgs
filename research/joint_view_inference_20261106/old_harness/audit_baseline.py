"""冻结模型真实预算推理审计；只读模型/生产目录。"""
from pathlib import Path
import json
import statistics
import sys
import time
import numpy as np
import torch

root=Path(__file__).parent
sys.path.insert(0,str(root/'source/python'))
from budget_viewer import BudgetScene
from utils import MiniCam,focal2fov
from renderer import render_prepared

torch.set_num_threads(2)
scene=BudgetScene('/home/prmi/zwj/realLOD/results/fr1_xyz_frozen_master_20261103_01/r20_40_60_80/training/13052_shutdown',
    '/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz')
c=scene.calibration
camera=lambda i:MiniCam(c['width'],c['height'],focal2fov(c['fx'],c['width']),focal2fov(c['fy'],c['height']),np.linalg.inv(scene.views[i]['pose']))
rows=[]
for ratio in [.01,.05,.1,.2,.4,.8]:
    for j in range(30):
        frame=[0,198,396,594,791][j%5]
        torch.cuda.synchronize();started=time.perf_counter()
        data,ids,stats=scene.renderer.prepare(camera(frame),ratio_budget=ratio)
        torch.cuda.synchronize();wall=(time.perf_counter()-started)*1000
        if j>=5:rows.append(dict(frame=frame,ratio=ratio,wall_ms=wall,**{k:v for k,v in stats.items() if k!='ratio'}))
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,with_stack=True) as profile:
    for i in [0,396,791]:
        with torch.profiler.record_function('BASELINE_VISIBLE40'):
            data,_,_=scene.renderer.prepare(camera(i),ratio_budget=.4)
        with torch.profiler.record_function('RASTERIZER'):
            render_prepared(camera(i),data,scene.background)
profile.export_chrome_trace(str(root/'baseline_trace.json'))
(root/'baseline_profile.txt').write_text(profile.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
summary=[]
for r in [.01,.05,.1,.2,.4,.8]:
    values=[row for row in rows if row['ratio']==r]
    summary.append(dict(ratio=r,mean_k=statistics.mean(row['active_gaussians'] for row in values),
        wall_ms=statistics.median(row['wall_ms'] for row in values),
        selector_ms=statistics.median(row['selector_ms'] for row in values),
        compensation_ms=statistics.median(row['compensation_ms'] for row in values)))
(root/'BASELINE_AUDIT.json').write_text(json.dumps(dict(summary=summary,rows=rows,
    model=str(scene.model),ratios=list(scene.renderer.knots),torch=torch.__version__,tf32=torch.backends.cuda.matmul.allow_tf32),indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
