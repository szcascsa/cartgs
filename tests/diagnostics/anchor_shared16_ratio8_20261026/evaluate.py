"""单进程复用模型/相机：Cached与same-K泛化，Full单独评价，不调参。"""
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from evo.core import sync
from evo.tools import file_interface

root=Path('/home/prmi/zwj/anchor-shared16-ratio8-20261026');repo=Path('/home/prmi/zwj/mycartgs');out=root/'validation'
sys.path.insert(0,str(repo/'python'))
from selector import load_selector,load_selector_metadata,compensated_attributes,mask_inactive_slots
from gaussian_model import GaussianModel
from renderer import prepare_render_data,_make_rasterizer
from utils import MiniCam,focal2fov
from diff_gaussian_rasterization import _C

torch.set_num_threads(2);torch.set_grad_enabled(False)
assert json.loads((out/'training_status.json').read_text())['stage']=='TRAINED'
protocol=json.loads((out/'preregistration.json').read_text());runtime=json.loads((out/'training_runtime.json').read_text())
data=Path('/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz')
rgb=np.loadtxt(data/'rgb.txt',dtype=str,comments='#');frames=protocol['evaluation_frames']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
models={};bank_union=set();hashes={};population={};eligible_counts={}
for name,info in runtime.items():
    cp=Path(info['checkpoint']);cloud=next((cp/'point_cloud').glob('iteration_*'))
    model=load_selector(str(cp/'selector.pt'),'cuda').requires_grad_(False)
    gaussian=GaussianModel(3);gaussian.load_ply(str(cloud/'point_cloud.ply'))
    for p in gaussian.__dict__.values():
        if isinstance(p,torch.Tensor):p.requires_grad_(False)
    meta=load_selector_metadata(str(cloud/'selector_metadata.ply'),'cuda')
    assert torch.equal(meta['maturity'],torch.ones_like(meta['maturity']))
    n=len(gaussian.get_xyz)
    cameras=json.loads((cp/'cameras.json').read_text());bank_union.update(c['img_name'] for c in cameras)
    gt=file_interface.read_tum_trajectory_file(str(data/'groundtruth.txt'))
    trajectory=file_interface.read_tum_trajectory_file(str(cp.parent.parent/'CameraTrajectory_TUM.txt'))
    gt,trajectory=sync.associate_trajectories(gt,trajectory,max_diff=.08)
    aligned=copy.deepcopy(trajectory);aligned.align(gt,correct_scale=False)
    ate=float(np.mean(np.sum((aligned.positions_xyz-gt.positions_xyz)**2,axis=1))**.5)
    eligible=~(model.anchors.slot_inactive.flatten()&(gaussian.get_opacity.flatten()<1/255))
    models[name]=dict(model=model,g=gaussian,meta=meta,intr=cameras[0],trajectory=trajectory,eligible=eligible,ate=ate)
    population[name]=n;eligible_counts[name]=int(eligible.sum())
    for path in [cp/'selector.pt',cloud/'point_cloud.ply',cloud/'selector_metadata.ply',cp/'cameras.json',cp/'selector_config.yaml']:
        hashes[str(path)]=sha(path)
nonbank=[i for i in frames if rgb[i,1] not in bank_union]
assert len(frames)==64 and len(nonbank)>=32
common_k={str(r):min(int(np.floor(r*min(population.values()))),min(eligible_counts.values())) for r in protocol['evaluation_ratios']}
assert min(common_k.values())>0
plan=dict(frames=frames,nonbank_frames=nonbank,populations=population,eligible_counts=eligible_counts,common_k=common_k,
    bank_union_paths=sorted(bank_union),model_sha256=hashes,poses='each final online map own estimated poses; same RGB IDs',
    same_k='pre-softmax logit difference - logit(cached uniform), stable row tie; query original ratio for all models; commonK floor(r*minN), cap common eligible',
    full='uncompensated Master;100% Selector is separate case',ratio_interpolation=protocol['interpolation'])
if (out/'evaluation_plan.json').exists():assert json.loads((out/'evaluation_plan.json').read_text())==plan
else:(out/'evaluation_plan.json').write_text(json.dumps(plan,indent=2))
metric_ssim=StructuralSimilarityIndexMeasure(data_range=1.).cuda()
metric_lpips=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda()
background=torch.zeros(3,device='cuda');rows=[];summaries=[];selection=[];sets={};parity=[]

def write(name,items):
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)

def metrics(image,truth):
    pred=image.clamp(0,1).unsqueeze(0);gt=truth.unsqueeze(0)
    result=dict(psnr=float(-10*torch.log10((pred-gt).square().mean())),ssim=float(metric_ssim(pred,gt)),
                lpips=float(metric_lpips(pred,gt)),photo=float(.8*(image-truth).abs().mean()+.2*(1-_C.fusedssim(.0001,.0009,image.contiguous(),truth).mean())))
    metric_ssim.reset();metric_lpips.reset()
    assert all(np.isfinite(v) for v in result.values());return result

for name,info in runtime.items():
    item=models[name];model=item['model'];g=item['g'];meta=item['meta'];n=population[name]
    intr=item['intr'];views=[]
    for i in frames:
        trajectory=item['trajectory'];j=int(np.argmin(abs(trajectory.timestamps-float(rgb[i,0]))))
        assert abs(trajectory.timestamps[j]-float(rgb[i,0]))<.08
        cam=MiniCam(intr['width'],intr['height'],focal2fov(intr['fx'],intr['width']),focal2fov(intr['fy'],intr['height']),np.linalg.inv(trajectory.poses_se3[j]))
        image=cv2.cvtColor(cv2.imread(str(data/rgb[i,1])),cv2.COLOR_BGR2RGB)
        views.append(dict(index=i,path=rgb[i,1],cam=cam,gt=torch.as_tensor(image,device='cuda').permute(2,0,1).float().contiguous()/255))
    cases=[('Full',1.,None,None)]
    for r in protocol['evaluation_ratios']:
        p,h=model.forward_selection_features(g.get_xyz,g.get_scaling,g.get_features_dc,g.get_opacity,torch.tensor([[r]],device='cuda'))
        logits=model.soft_net(model.local_emd[3](model.local_emd[2](h)))
        assert torch.equal(p,logits.softmax(-1)[:,1])
        masked=mask_inactive_slots(model,p,g.get_opacity)
        cached=(meta['cached_uniform']<masked)
        margin=((logits[:,1]-logits[:,0]).double()-torch.logit(meta['cached_uniform'].double())).cpu().numpy()
        margin[~item['eligible'].cpu().numpy()]=-np.inf
        assert not np.isnan(margin).any()
        order=np.lexsort((np.arange(n),-margin))
        actual=int(cached.sum());reproduced=np.sort(order[:actual])
        assert np.array_equal(reproduced,torch.nonzero(cached).flatten().cpu().numpy())
        k=common_k[str(r)];same=torch.zeros_like(cached);same[torch.as_tensor(order[:k].copy(),device='cuda')]=True
        assert item['eligible'][same].all() and int(same.sum())==k
        cases.extend([('cached',r,cached,h),('same_k',r,same,h)])
        selection.append(dict(model=name,ratio=r,population=n,eligible=eligible_counts[name],cached_k=actual,same_k=k,
            cached_same_k_intersection=int((cached&same).sum()),p_mean=float(p.mean()),p_min=float(p.min()),p_max=float(p.max()),
            p_zero=int((p==0).sum()),p_one=int((p==1).sum()),cached_margin_reproduction=True))
    for policy,r,mask,h in cases:
        label=f'{policy}_{r:g}'
        attributes=compensated_attributes(model,g,mask.float(),r,h) if mask is not None else None
        prepared=prepare_render_data(g,mask,attributes);inputs=prepared['inputs'];k=len(inputs['means3D'])
        if mask is not None:sets[name+'_'+label]=torch.nonzero(mask).flatten().cpu().numpy()
        case=[]
        for vi,v in enumerate(views):
            torch.cuda.synchronize();start=time.perf_counter()
            if k:
                image,radii=_make_rasterizer(v['cam'],g.active_sh_degree,background,1.,inference_only=True)(**inputs)
            else:
                image=background[:,None,None].expand_as(v['gt']).contiguous();radii=torch.empty(0,device='cuda')
            torch.cuda.synchronize();render_ms=(time.perf_counter()-start)*1000
            if vi==0:
                other,other_radii=_make_rasterizer(v['cam'],g.active_sh_degree,background,1.)(**inputs) if k else (image,radii)
                error=float((other-image).abs().max());assert error<5e-6 and torch.equal(radii,other_radii)
                parity.append(dict(model=name,policy=policy,ratio=r,error=error,radii_exact=True))
            case.append(dict(model=name,method=info['method'],seed=info['seed'],policy=policy,ratio=r,
                gt_index=v['index'],path=v['path'],nonbank=v['index'] in nonbank,population=n,selected=k,
                visible=int((radii>0).sum()),opacity_effective_proxy=int(((radii>0)&(inputs['opacities'].flatten()>=1/255)).sum()),
                render_ms=render_ms,**metrics(image,v['gt'])))
            if info['seed']==2026 and policy=='cached' and r in [.01,.05,.1,.15,.5,1.] and vi in [16,32]:
                destination=root/'preview';destination.mkdir(exist_ok=True)
                cv2.imwrite(str(destination/f'{name}_r{r:g}_view{vi}.png'),cv2.cvtColor((image.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8),cv2.COLOR_RGB2BGR))
        rows.extend(case)
        for split in ['all64','nonbank']:
            subset=[x for x in case if split=='all64' or x['nonbank']]
            summaries.append(dict(model=name,method=info['method'],seed=info['seed'],policy=policy,ratio=r,split=split,views=len(subset),
                population=n,selected=k,ate_rmse=item['ate'],**{key:float(np.mean([x[key] for x in subset])) for key in ['psnr','ssim','lpips','photo','visible','opacity_effective_proxy','render_ms']}))
        write('quality_per_view.csv',rows);write('quality_summary.csv',summaries);write('selection_statistics.csv',selection)
        (out/'evaluation_status.json').write_text(json.dumps(dict(stage='EVALUATING',model=name,case=label,total_view_rows=len(rows))))
        print('QUALITY',name,label,summaries[-1]['psnr'],flush=True)
    del views
np.savez_compressed(out/'selected_sets.npz',**sets)
paired=[]
for candidate,reference in [('shared16_ratio8','shared16'),('shared16_ratio8','shared8_ratio8'),('shared16_ratio8','r3_independent14')]:
    for seed in protocol['seeds']:
        for policy in ['Full','cached','same_k']:
            for r in ([1.] if policy=='Full' else protocol['evaluation_ratios']):
                for split in ['all64','nonbank']:
                    a=[x for x in rows if x['method']==candidate and x['seed']==seed and x['policy']==policy and x['ratio']==r and (split=='all64' or x['nonbank'])]
                    b=[x for x in rows if x['method']==reference and x['seed']==seed and x['policy']==policy and x['ratio']==r and (split=='all64' or x['nonbank'])]
                    assert [x['path'] for x in a]==[x['path'] for x in b]
                    for metric,sign in [('psnr',1),('ssim',1),('lpips',-1),('photo',-1)]:
                        d=np.array([x[metric]-y[metric] for x,y in zip(a,b)])
                        paired.append(dict(seed=seed,policy=policy,ratio=r,split=split,metric=metric,comparison=candidate+'-'+reference,
                            mean=float(d.mean()),median=float(np.median(d)),p10=float(np.quantile(d,.1)),p90=float(np.quantile(d,.9)),improved_fraction=float((d*sign>0).mean())))
write('paired_statistics.csv',paired)
assert all(sha(path)==h for path,h in hashes.items())
(out/'quality_complete.json').write_text(json.dumps(dict(passed=True,rows=len(rows),summaries=len(summaries),
    nonbank=len(nonbank),models_unchanged=True,parity=parity,ratios=protocol['evaluation_ratios'],tuned_on_eval=False),indent=2))
print('ALL_QUALITY_COMPLETE',flush=True)
