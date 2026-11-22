"""四次真实SLAM的预定64视图、四档Cached和common-K质量；不用于挑选参数。"""
import csv
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from evo.core import sync
from evo.tools import file_interface

R=Path('/home/prmi/zwj/anchor-selector-r3-cuda-optimization-20261022');S=R/'source';D=Path('/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz')
sys.path.insert(0,str(S/'python'))
from selector import load_selector,load_selector_metadata,compensated_attributes,mask_inactive_slots
from gaussian_model import GaussianModel
from renderer import prepare_render_data,_make_rasterizer
from utils import MiniCam,focal2fov
from diff_gaussian_rasterization import _C

torch.set_num_threads(2);torch.set_grad_enabled(False)
protocol=json.loads(Path('/home/prmi/zwj/private-compensation-online-20261022/validation/preregistration.json').read_text())
frames=protocol['evaluation_frames'];rgb=np.loadtxt(D/'rgb.txt',dtype=str,comments='#');ratios=[.01,.05,.1,.15]
models={};bank_union=set();rows=[]
for name in ['online_legacy_a','online_fused_a','online_fused_b','online_legacy_b']:
    cp=next((R/name).glob('*_shutdown/ply'));cloud=next((cp/'point_cloud').glob('iteration_*'))
    m=load_selector(str(cp/'selector.pt'),'cuda').requires_grad_(False);g=GaussianModel(3);g.load_ply(str(cloud/'point_cloud.ply'))
    metadata=load_selector_metadata(str(cloud/'selector_metadata.ply'),'cuda');cameras=json.loads((cp/'cameras.json').read_text())
    bank_union.update(c['img_name'] for c in cameras)
    gt=file_interface.read_tum_trajectory_file(str(D/'groundtruth.txt'));trajectory=file_interface.read_tum_trajectory_file(str(R/name/'CameraTrajectory_TUM.txt'))
    gt,trajectory=sync.associate_trajectories(gt,trajectory,max_diff=.08)
    eligible=~(m.anchors.slot_inactive.flatten()&(g.get_opacity.flatten()<1/255))
    models[name]=dict(cp=cp,model=m,g=g,meta=metadata,intr=cameras[0],trajectory=trajectory,eligible=eligible,N=len(g.get_xyz))
minimum=min(x['N'] for x in models.values());eligible_count=min(int(x['eligible'].sum()) for x in models.values())
common={r:min(int(np.floor(r*minimum)),eligible_count) for r in ratios}
(R/'online_quality_protocol.json').write_text(json.dumps(dict(frames=frames,ratios=ratios,common_k=common,
    original_frame_protocol='/home/prmi/zwj/private-compensation-online-20261022/validation/preregistration.json',
    bank_union=sorted(bank_union),models={n:dict(N=x['N'],checkpoint=str(x['cp'])) for n,x in models.items()},
    poses='each online map own estimated pose; same RGB IDs; topology cannot be forced equal in asynchronous SLAM',tuned_on_eval=False),indent=2))
ssim=StructuralSimilarityIndexMeasure(data_range=1.).cuda();lpips=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda();bg=torch.zeros(3,device='cuda')
def write(name,items):
    with (R/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
for name,x in models.items():
    m,g,meta=x['model'],x['g'],x['meta'];views=[];intr=x['intr']
    for index in frames:
        trajectory=x['trajectory'];j=int(np.argmin(abs(trajectory.timestamps-float(rgb[index,0]))))
        assert abs(trajectory.timestamps[j]-float(rgb[index,0]))<.08
        cam=MiniCam(intr['width'],intr['height'],focal2fov(intr['fx'],intr['width']),focal2fov(intr['fy'],intr['height']),np.linalg.inv(trajectory.poses_se3[j]))
        image=cv2.cvtColor(cv2.imread(str(D/rgb[index,1])),cv2.COLOR_BGR2RGB)
        views.append((index,cam,torch.as_tensor(image,device='cuda').permute(2,0,1).float().contiguous()/255))
    cases=[('Full',1.,None,None)]
    for r in ratios:
        p,h=m.forward_selection_features(g.get_xyz,g.get_scaling,g.get_features_dc,g.get_opacity,torch.tensor([[r]],device='cuda'))
        logits=m.soft_net(m.local_emd[3](m.local_emd[2](h)));cached=meta['cached_uniform']<mask_inactive_slots(m,p,g.get_opacity)
        margin=((logits[:,1]-logits[:,0]).double()-torch.logit(meta['cached_uniform'].double())).cpu().numpy();margin[~x['eligible'].cpu().numpy()]=-np.inf
        order=np.lexsort((np.arange(x['N']),-margin));same=torch.zeros_like(cached);same[torch.as_tensor(order[:common[r]].copy(),device='cuda')]=True
        assert int(same.sum())==common[r];cases.extend([('cached',r,cached,h),('same_k',r,same,h)])
    for policy,r,mask,h in cases:
        attrs=compensated_attributes(m,g,mask.float(),r,h) if mask is not None else None
        inputs=prepare_render_data(g,mask,attrs)['inputs'];k=len(inputs['means3D'])
        for index,cam,truth in views:
            image,radii=_make_rasterizer(cam,g.active_sh_degree,bg,1.,inference_only=True)(**inputs)
            pred=image.clamp(0,1).unsqueeze(0);target=truth.unsqueeze(0)
            rows.append(dict(run=name,policy=policy,ratio=r,index=index,nonbank=rgb[index,1] not in bank_union,N=x['N'],K=k,
                visible=int((radii>0).sum()),effective_proxy=int(((radii>0)&(inputs['opacities'].flatten()>=1/255)).sum()),
                psnr=float(-10*torch.log10((pred-target).square().mean())),ssim=float(ssim(pred,target)),lpips=float(lpips(pred,target)),
                photo=float(.8*(image-truth).abs().mean()+.2*(1-_C.fusedssim(.0001,.0009,image.contiguous(),truth).mean()))))
            ssim.reset();lpips.reset()
        write('online_quality_per_view.csv',rows)
        print('QUALITY',name,policy,r,flush=True)
summary=[]
for name in models:
    for policy,r in [('Full',1.)]+[(p,r) for p in ['cached','same_k'] for r in ratios]:
        for split in ['all64','nonbank']:
            subset=[x for x in rows if x['run']==name and x['policy']==policy and x['ratio']==r and (split=='all64' or x['nonbank'])]
            summary.append(dict(run=name,policy=policy,ratio=r,split=split,views=len(subset),N=subset[0]['N'],K=subset[0]['K'],
                **{metric:float(np.mean([x[metric] for x in subset])) for metric in ['psnr','ssim','lpips','photo','visible','effective_proxy']}))
write('online_quality_summary.csv',summary)
(R/'online_quality_complete.json').write_text(json.dumps(dict(passed=True,per_view_rows=len(rows),summaries=len(summary),
    limitation='asynchronous maps differ; this is repeat-envelope evidence, not a matched topology/clock causal quality test'),indent=2))
