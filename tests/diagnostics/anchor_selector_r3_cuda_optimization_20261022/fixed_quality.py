"""固定camera真实重放图像指标，独立于Profiler和正式计时。"""
import csv
from pathlib import Path
import sys
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
R=Path('/home/prmi/zwj/anchor-selector-r3-cuda-optimization-20261022');sys.path.insert(0,str(R/'source/python'))
from diff_gaussian_rasterization import _C
torch.set_num_threads(2);torch.set_grad_enabled(False)
ssim=StructuralSimilarityIndexMeasure(data_range=1.).cuda();lpips=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda();rows=[]
for run in ['legacy_a','legacy_b','adam','adam_decode','fused_a','fused_b']:
    for case in ['Full','lod_0.010000','lod_0.050000','lod_0.100000','lod_0.150000']:
        a=torch.jit.load(str(R/run/(case+'_parity.pt')),map_location='cuda').state_dict();image=a['render'];truth=a['gt'];pred=image.clamp(0,1).unsqueeze(0);target=truth.unsqueeze(0)
        rows.append(dict(run=run,case=case,N=a['owner'].numel(),K=a['selected_ids'].numel() if 'selected_ids' in a else a['owner'].numel(),
            visible=int((a['radii']>0).sum()),psnr=float(-10*torch.log10((pred-target).square().mean())),ssim=float(ssim(pred,target)),lpips=float(lpips(pred,target)),
            photo=float(.8*(image-truth).abs().mean()+.2*(1-_C.fusedssim(.0001,.0009,image.contiguous(),truth).mean())),total_loss=float(a['loss'])))
        ssim.reset();lpips.reset()
with (R/'fixed_quality.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('PASS Fixed-view PSNR/SSIM/LPIPS/photo metrics',flush=True)
