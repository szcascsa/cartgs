"""相同固定checkpoint/帧/K的渲染质量；不用于选择实现或训练配置。"""
import os
from pathlib import Path
import csv
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
R=Path(os.environ.get('CARTGS_PHASE2_RESULTS','/home/prmi/zwj/anchor-selector-r3-cuda-phase2-20261023'));torch.set_num_threads(2);torch.set_grad_enabled(False)
ssim=StructuralSimilarityIndexMeasure(data_range=1.).cuda();lpips=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda();rows=[]
for resolution in ['1.0','0.5']:
 for variant in ['A','B','C','D','E_workspace','E']:
  for case in ['Full','lod_0.010000','lod_0.050000','lod_0.100000','lod_0.150000']:
    a=torch.jit.load(str(R/f'bench_{resolution}_0_{variant}'/f'{case}_parity.pt'),map_location='cuda').state_dict();image=a['render'];truth=a['gt'];pred=image.clamp(0,1).unsqueeze(0);target=truth.unsqueeze(0)
    rows.append(dict(resolution=float(resolution),variant=variant,case=case,N=a['owner'].numel(),K=a['selected_ids'].numel() if 'selected_ids' in a else a['owner'].numel(),visible=int((a['radii']>0).sum()),psnr=float(-10*(pred-target).square().mean().log10()),psnr_unclamped=float(-10*(image-truth).square().mean().log10()),ssim=float(ssim(pred,target)),lpips=float(lpips(pred,target)),photo=float(a['photo']),l1=float(a['l1']),loss_ssim=float(a['ssim']),budget=float(a.get('budget',torch.tensor(0.)))))
    ssim.reset();lpips.reset()
with (R/'fixed_quality_metrics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('QUALITY_PASS',flush=True)
