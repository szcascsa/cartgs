"""冻结模型停用验证；same-K 的统一 logit 平移仅作离线诊断，不改生产采样。"""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
import cv2

root = Path('/home/prmi/zwj/anchor-slot-deactivation-20261021')
old = Path('/home/prmi/zwj/octree-anchor-pruning-20261021')
repo = old/'source'
out = root/'frozen'; out.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(repo/'python'))
from evo.tools import settings
settings.SETTINGS.plot_backend = 'Agg'
import run as evaluator
from renderer import prepare_render_data, render_prepared
from selector import load_selector, load_selector_metadata, build_selector_mask
from transform_field import load_transform_field, transformed_attributes

run = old/'smoke_seed2026'
checkpoint = run/'10000_shutdown/ply'
cloud = checkpoint/'point_cloud/iteration_10000'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths = [checkpoint/'selector.pt', checkpoint/'transform_field.pt', cloud/'point_cloud.ply',
         cloud/'selector_metadata.ply', cloud/'anchor_pruning_state.pt']
hashes = {str(p): sha(p) for p in paths}
available = json.loads(Path('/home/prmi/zwj/octree-explicit-20261021/validation/common_frames.json').read_text())['gt_indices']
frames = [available[i] for i in np.linspace(0, len(available)-1, 64, dtype=int)]
protocol = dict(scene='fr1_xyz', seed=2026, ratios=[.01,.05,.1,.15], frames=frames,
    model_sha256=hashes, population=41730, cutoff=1/255,
    history='latest closed window with T > 0.8 * check_interval; every visit below cutoff; current below cutoff',
    controls=['cached_baseline','drop_only','cached_logit_shift_same_k'],
    redistribution='rank logit(p)-logit(saved_uniform) on eligible rows; K=baseline actual cached count; diagnostic only',
    gates=dict(drop_only_max_abs_image=1e-6, same_k_per_ratio_min_delta_psnr=-.02,
               same_k_per_ratio_min_delta_ssim=-.0005, same_k_per_ratio_max_delta_lpips=.001,
               high_ratio_mean_min_delta_psnr=.05),
    frozen=True, training=False, no_quality_tuning=True)
(out/'preregistration.json').write_text(json.dumps(protocol,indent=2))
def write(name, rows):
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def arrays(path):
    return {k:v.detach().cpu().numpy() for k,v in torch.jit.load(str(path),map_location='cpu').named_buffers()}

g=evaluator.GaussianModel(3); g.load_ply(str(cloud/'point_cloud.ply'))
model=load_selector(str(checkpoint/'selector.pt'),'cuda')
meta=load_selector_metadata(str(cloud/'selector_metadata.ply'),'cuda')
config=evaluator.load_selector_config(str(checkpoint/'selector_config.yaml'))
assert not config['protection_enabled'] and (meta['maturity']==1).all()
uid=model.anchors.block_uids.cpu().numpy(); history={}
for path in sorted((run/'anchor_pruning').glob('window_*.pt'),key=lambda p:int(p.stem.split('_')[-1])):
    d=arrays(path)
    for j,u in enumerate(d['block_uids']):
        if d['anchor_visits'][j] > .8*float(d['check_interval']):
            history[int(u)] = (d['slot_below_cutoff_count'][j]==d['anchor_visits'][j])
qualified=np.stack([history.get(int(u),np.zeros(10,bool)) for u in uid]).reshape(-1)
cut=g.get_opacity.flatten()<1/255
blocked=torch.as_tensor(qualified,device='cuda') & cut
np.savez_compressed(out/'eligibility.npz',uids=uid,history_qualified=qualified,blocked=blocked.cpu().numpy())

dataset='/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz'
rgb,times=evaluator.loadTUM(dataset)
truth=evaluator.file_interface.read_tum_trajectory_file(dataset+'/groundtruth.txt')
estimate=evaluator.file_interface.read_tum_trajectory_file(str(run/'CameraTrajectory_TUM.txt'))
_,trajectory=evaluator.sync.associate_trajectories(truth,estimate,max_diff=.08)
pairs=evaluator.associate_frames(trajectory.timestamps,times)
pairs=[p for p in pairs if p[1] in set(frames)]
assert len(pairs)==len(frames) and {j for _,j in pairs}==set(frames)
camera=json.loads((checkpoint/'cameras.json').read_text())[0]
fx=evaluator.focal2fov(camera['fx'],camera['width']);fy=evaluator.focal2fov(camera['fy'],camera['height'])
bg=torch.zeros(3,device='cuda'); tf=load_transform_field(str(checkpoint),'cuda')
quality=[];per_view=[];image_errors=[];selections=[]
with torch.no_grad():
    for ratio in protocol['ratios']:
        baseline,stats=build_selector_mask(model,g,meta,ratio,config,config['temperature'],False)
        base=baseline.bool(); k=int(base.sum())
        p=model(g.get_xyz,None,g.get_scaling,torch.tensor([[ratio]],device='cuda'),
                tau=config['temperature'],sh_dc=g.get_features[:,:1],opacity=g.get_opacity)
        assert torch.equal(meta['cached_uniform']<p,base)
        margin=(torch.logit(p.double())-torch.logit(meta['cached_uniform'].double())).cpu().numpy()
        candidates=np.flatnonzero(~blocked.cpu().numpy())
        order=candidates[np.lexsort((candidates,-margin[candidates]))]
        redistributed=torch.zeros_like(base);redistributed[torch.as_tensor(order[:k].copy(),device='cuda')]=True
        assert int(redistributed.sum())==k and not (redistributed & blocked).any()
        assert (base & ~blocked & ~redistributed).sum()==0
        # Compute TF once so retained rows have exactly the same geometry in all arms.
        attributes=transformed_attributes(tf,g,torch.ones_like(base),meta['maturity'],ratio)
        masks=dict(cached_baseline=base,drop_only=base & ~blocked,cached_logit_shift_same_k=redistributed)
        data={name:prepare_render_data(g,mask,attributes) for name,mask in masks.items()}
        for name,mask in masks.items():
            np.save(out/(str(ratio)+'_'+name+'_rows.npy'),torch.nonzero(mask).flatten().cpu().numpy())
            selections.append(dict(ratio=ratio,arm=name,population=len(base),selected=int(mask.sum()),
                blocked_population=int(blocked.sum()),blocked_selected=int((mask & blocked).sum()),
                below_cutoff_selected=int((mask & cut).sum()),baseline_actual_k=k))
        for index,(i,j) in enumerate(pairs):
            cam=evaluator.MiniCam(camera['width'],camera['height'],fx,fy,np.linalg.inv(trajectory.poses_se3[i]))
            gt=torch.from_numpy(cv2.cvtColor(cv2.imread(rgb[j]),cv2.COLOR_BGR2RGB)).to('cuda').permute(2,0,1)[None]/255.
            images={name:render_prepared(cam,d,bg)['render'] for name,d in data.items()}
            image_errors.append(dict(ratio=ratio,gt_index=j,max_abs=float((images['cached_baseline']-images['drop_only']).abs().max())))
            for name,img in images.items():
                value=img.clamp(0,1)[None]
                per_view.append(dict(ratio=ratio,arm=name,gt_index=j,
                    psnr=evaluator.calc_psnr(value,gt).item(),ssim=evaluator.calc_ssim(value,gt).item(),
                    lpips=evaluator.calc_lpips(value,gt).item()))
            if index%16==0:print('progress',ratio,index,flush=True)
        for name in masks:
            rows=[r for r in per_view if r['ratio']==ratio and r['arm']==name]
            quality.append(dict(ratio=ratio,arm=name,views=len(rows),selected=int(masks[name].sum()),
                **{m:float(np.mean([r[m] for r in rows])) for m in ['psnr','ssim','lpips']}))
        write('quality_summary.csv',quality);write('quality_per_view.csv',per_view)
        print('QUALITY',quality[-3:],flush=True)
write('selections.csv',selections);write('drop_image_errors.csv',image_errors)
paired=[]
for ratio in protocol['ratios']:
    for metric,sign in [('psnr',1),('ssim',1),('lpips',-1)]:
        a=np.array([r[metric] for r in per_view if r['ratio']==ratio and r['arm']=='cached_logit_shift_same_k'])
        b=np.array([r[metric] for r in per_view if r['ratio']==ratio and r['arm']=='cached_baseline'])
        delta=a-b
        paired.append(dict(ratio=ratio,metric=metric,mean=float(delta.mean()),median=float(np.median(delta)),
            p10=float(np.quantile(delta,.1)),p90=float(np.quantile(delta,.9)),improved_fraction=float((sign*delta>0).mean())))
write('paired_statistics.csv',paired)
gates=dict(drop_only_zero_contribution=max(r['max_abs'] for r in image_errors)<=1e-6,
    same_k_all_ratios_no_material_regression=all(r['mean']>=(-.02 if r['metric']=='psnr' else -.0005)
        if r['metric']!='lpips' else r['mean']<=.001 for r in paired),
    high_ratio_mean_quality_improves=np.mean([r['mean'] for r in paired if r['metric']=='psnr' and r['ratio']>=.05])>=.05,
    frozen_model_sha_unchanged=all(sha(p)==h for p,h in hashes.items()))
summary=dict(gates={k:bool(v) for k,v in gates.items()},passed=bool(all(gates.values())),
    blocked=int(blocked.sum()),population=len(blocked),views=len(frames),same_k_diagnostic_only=True,
    production_redistribution_requires_training=True,quality=quality,paired=paired)
(out/'gate_result.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2),flush=True)
