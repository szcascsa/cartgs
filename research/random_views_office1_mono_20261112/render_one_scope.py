"""复用baseline评价路径，仅渲染预选10帧并保存无损PNG和真实使用位姿。"""
from pathlib import Path
import csv
import json
import os
import runpy
import sys

root=Path(sys.argv[1]);scope=sys.argv[2]
metadata=json.loads((root/'EXPERIMENT.json').read_text())
batch=Path(metadata['batch']);model=Path(metadata['model'])
output=root/({'full':'full_0','global':'global_0.3','view_ratio':'local_0.3'}[scope])
sys.path[:0]=[str(batch/'source/python'),str(batch/'evaluation_runtime')]
import numpy as np
import torch
import cv2
from evo.tools.settings import SETTINGS
SETTINGS.plot_backend='Agg'
torch.manual_seed(0);torch.cuda.manual_seed_all(0)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

if scope=='view_ratio':
    import anchor_frustum
    from anchor_budget import FixedBudgetRenderer

    class LocalRatioRenderer:
        def __init__(self,selector,gaussians,ratio,config,temperature,protection):
            assert int(selector.joint_view_conditioning_schema)==2
            self.ratio=ratio
            self.renderer=FixedBudgetRenderer(selector,gaussians,
                dict(config,temperature=temperature,protection_enabled=protection),seed=0,backend='pytorch')

        def prepare(self,camera):
            data,ids,stats=self.renderer.prepare(camera,ratio_budget=self.ratio,view_conditioned=True)
            assert stats['view_conditioned'] and len(ids)==min(stats['requested_budget'],stats['eligible_candidates'])
            stats['legacy_quantile_budget']=False
            return data,ids,stats

    anchor_frustum.AnchorFrustumRenderer=LocalRatioRenderer

# 保留原评价的像素与计算，仅将落盘格式改为PNG；GT只保存一份。
original_imwrite=cv2.imwrite
def save_png(path,pixels,*args,**kwargs):
    target=Path(path)
    if target.parent.name=='gt' and scope!='full':return True
    if target.parent.name in ['gt','image']:target=target.with_suffix('.png')
    assert original_imwrite(str(target),pixels,*args,**kwargs),str(target)
    return True
cv2.imwrite=save_png

script=batch/'source/python/run_metrics_only.py'
sys.argv=[str(script),str(model),metadata['data'],'--shutdown-model',metadata['shutdown'],
    '--output_path',str(output),'--correct_scale','--evaluation-frame-set',str(root/'FRAME_SET.json')]
if scope!='full':
    sys.argv+=['--selector',metadata['selector'],'--selector-ratio','0.3',
        '--selector-config',metadata['selector_config']]
if scope=='view_ratio':sys.argv+=['--selector-anchor-frustum']
values=runpy.run_path(str(script),run_name='__main__')
rows=[]
frustum=values['frustum_stats']
for index,(trajectory_index,rgb_index) in enumerate(values['associations']):
    c2w=np.asarray(values['poses'][trajectory_index])
    stats=frustum[index] if scope=='view_ratio' else values['selector_stats']
    row=dict(scope=scope,gt_index=int(rgb_index),trajectory_index=int(trajectory_index),
        timestamp=float(values['tstamp'][trajectory_index]),rgb_path=values['gt_color_paths'][rgb_index],
        c2w=c2w.tolist(),w2c=np.linalg.inv(c2w).tolist(),
        psnr=float(values['psnr_list'][index]),ssim=float(values['ssim_list'][index]),
        lpips=float(values['lpips_list'][index]),K=int(stats['active_gaussians']),
        candidate_N=int(stats['candidate_gaussians']) if scope=='view_ratio' else None,
        requested_budget=int(stats['requested_budget']) if scope=='view_ratio' else None,
        image=str(output/'image'/Path(values['gt_color_paths'][rgb_index]).with_suffix('.png').name))
    assert Path(row['image']).is_file()
    rows.append(row)
assert len(rows)==10 and [r['gt_index'] for r in rows]==metadata['gt_indices']
(output/'SELECTED_FRAME_RESULTS.json').write_text(json.dumps(rows,indent=2)+'\n')
print('RENDER_PASS',scope,len(rows),flush=True)
