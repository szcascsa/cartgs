"""一次停用训练质量对照；先锁共同视图、绝对 K，再运行现有 evaluator。"""
import csv
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import numpy as np
import torch

root=Path('/home/prmi/zwj/anchor-slot-deactivation-20261021')
repo=root/'source';out=root/'validation'
sys.path.insert(0,str(repo/'python'))
from evo.tools import settings
settings.SETTINGS.plot_backend='Agg'
import selector

if len(sys.argv)>1 and sys.argv[1]=='entry':
    policy=sys.argv.pop(2);sys.argv.pop(1)
    dest=Path(sys.argv[sys.argv.index('--output_path')+1])
    original=selector.build_selector_mask
    @torch.no_grad()
    def mask(model,g,metadata,ratio,config,temperature=1.,protection_enabled=True):
        selected,stats=original(model,g,metadata,ratio,config,temperature,protection_enabled)
        if policy=='same_k':
            k=json.loads((out/'evaluation_plan.json').read_text())['common_k'][str(ratio)]
            p=model(g.get_xyz,None,g.get_scaling,torch.tensor([[ratio]],device='cuda'),tau=temperature,
                    sh_dc=g.get_features[:,:1],opacity=g.get_opacity)
            p=selector.mask_inactive_slots(model,p,g.get_opacity)
            margin=(torch.logit(p.double())-torch.logit(metadata['cached_uniform'].double())).cpu().numpy()
            order=np.lexsort((np.arange(len(p)),-margin))
            # An inactive row can never be used to fill the diagnostic budget.
            assert not np.isnan(margin).any() and (p[torch.as_tensor(order[:k].copy(),device='cuda')]>0).all()
            selected=torch.zeros_like(selected);selected[torch.as_tensor(order[:k].copy(),device='cuda')]=1
            stats.update(active_gaussians=k,actual_active=k,common_absolute_k=k)
        stats['selection_policy']=policy
        rows=torch.nonzero(selected).flatten().cpu().numpy()
        np.savez_compressed(dest/'selected_rows.npz',rows=rows)
        alpha=g.get_opacity.flatten()
        stats['selected_below_cutoff']=int(((alpha<1/255)&selected.bool()).sum())
        stats['selected_below_002']=int(((alpha<.02)&selected.bool()).sum())
        return selected,stats
    selector.build_selector_mask=mask
    # Record exactly the existing native fused-SSIM photo, on unclamped output.
    import renderer
    import cv2
    from diff_gaussian_rasterization import _C
    render=renderer.render_prepared;photos=[]
    plan=json.loads((out/'evaluation_plan.json').read_text())
    images=np.loadtxt('/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz/rgb.txt',dtype=str,comments='#')[:,1]
    def wrapped(cam,data,bg,scaling_modifier=1.,measure_time=False):
        result=render(cam,data,bg,scaling_modifier,measure_time)
        if measure_time:
            j=plan['frames'][len(photos)]
            gt=cv2.cvtColor(cv2.imread('/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz/'+images[j]),cv2.COLOR_BGR2RGB)
            gt=torch.as_tensor(gt,device='cuda').permute(2,0,1).float().contiguous()/255.
            prediction=result['render'].contiguous()
            photo=.8*(prediction-gt).abs().mean()+.2*(1-_C.fusedssim(.0001,.0009,prediction,gt).mean())
            photos.append(float(photo))
        return result
    renderer.render_prepared=wrapped
    runpy.run_path(str(repo/'python/run.py'),run_name='__main__')
    assert len(photos)==len(plan['frames'])
    np.savetxt(dest/'photo.txt',photos)
    raise SystemExit()

from evo.core import sync
from evo.tools import file_interface
from plyfile import PlyData
baseline=Path('/home/prmi/zwj/octree-anchor-pruning-20261021/smoke_seed2026')
jobs={'baseline':baseline,'enabled':root/'enabled_seed2026'}
dataset=Path('/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz')
rgb=np.loadtxt(dataset/'rgb.txt',dtype=str,comments='#')[:,0].astype(float)
covered=[];population=[];hashes={}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for run in jobs.values():
    gt=file_interface.read_tum_trajectory_file(str(dataset/'groundtruth.txt'))
    est=file_interface.read_tum_trajectory_file(str(run/'CameraTrajectory_TUM.txt'))
    _,trajectory=sync.associate_trajectories(gt,est,max_diff=.08)
    ids=[int(np.argmin(abs(rgb-t))) for t in trajectory.timestamps if abs(rgb-t).min()<.08]
    assert len(ids)==len(set(ids));covered.append(set(ids))
    checkpoint=run/'10000_shutdown/ply';cloud=checkpoint/'point_cloud/iteration_10000'
    population.append(len(PlyData.read(cloud/'point_cloud.ply')['vertex']))
    for p in [checkpoint/'selector.pt',checkpoint/'transform_field.pt',cloud/'point_cloud.ply',cloud/'selector_metadata.ply']:
        hashes[str(p)]=sha(p)
available=sorted(set.intersection(*covered));assert len(available)>=700
frames=[available[i] for i in np.linspace(0,len(available)-1,64,dtype=int)]
plan=dict(frames=frames,populations=population,model_sha256=hashes,
    common_k={str(r):int(np.floor(r*min(population))) for r in [.01,.05,.1,.15]},
    policy='64 uniform temporal common associated frames; no quality filtering',
    same_k='cached logit margin order, source-row tie; diagnostic only',
    poses='each online model own estimated pose; asynchronous trajectories may differ')
(out/'evaluation_plan.json').write_text(json.dumps(plan,indent=2))
frames_path=out/'common_frames.json';frames_path.write_text(json.dumps(dict(gt_indices=frames),indent=2))
env=dict(os.environ,OMP_NUM_THREADS='2',MPLBACKEND='Agg',QT_QPA_PLATFORM='offscreen')
summary=[];views=[];metrics=['psnr','ssim','lpips','photo']
def write(name,rows):
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for name,run in jobs.items():
    checkpoint=run/'10000_shutdown/ply'
    for policy in ['cached_bernoulli','same_k']:
        levels=([('Full',None)] if policy=='cached_bernoulli' else [])+[(str(int(100*r)),r) for r in [.01,.05,.1,.15]]
        for level,ratio in levels:
            dest=root/'quality'/name/policy/level;dest.mkdir(parents=True,exist_ok=False)
            command=[sys.executable,__file__,'entry',policy,str(run),str(dataset),'--output_path',str(dest),
                '--evaluation-frame-set',str(frames_path),'--no-save-images']
            if ratio is not None:command+=['--selector',str(checkpoint/'selector.pt'),'--selector-ratio',str(ratio),
                '--selector-config',str(checkpoint/'selector_config.yaml')]
            with (dest/'evaluation.log').open('w') as f:
                subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            assert json.loads((dest/'evaluation_frames.json').read_text())['gt_indices']==frames
            s=json.loads((dest/'selector_stats.json').read_text())
            values={m:np.loadtxt(dest/(m+'.txt')) for m in metrics}
            assert all(v.shape==(64,) and np.isfinite(v).all() for v in values.values())
            row=dict(method=name,policy=policy,level=level,ratio=ratio,selected=s['active_gaussians'],
                total=s['total_gaussians'],inactive=s.get('inactive_slots',0),
                selected_below_cutoff=s.get('selected_below_cutoff',None),
                expected=s.get('expected_active',None),target=s.get('target_active',None),
                **{m:float(v.mean()) for m,v in values.items()})
            summary.append(row)
            for i,j in enumerate(frames):views.append(dict(method=name,policy=policy,level=level,gt_index=j,
                **{m:float(v[i]) for m,v in values.items()}))
            write('quality_summary.csv',summary);write('quality_per_view.csv',views)
            print('QUALITY',row,flush=True)
paired=[]
for policy in ['cached_bernoulli','same_k']:
    for level in (['Full','1','5','10','15'] if policy=='cached_bernoulli' else ['1','5','10','15']):
        for metric,sign in [('psnr',1),('ssim',1),('lpips',-1),('photo',-1)]:
            values={name:np.array([r[metric] for r in views if r['method']==name and r['policy']==policy and r['level']==level]) for name in jobs}
            d=values['enabled']-values['baseline']
            paired.append(dict(policy=policy,level=level,metric=metric,mean=float(d.mean()),median=float(np.median(d)),
                p10=float(np.quantile(d,.1)),p90=float(np.quantile(d,.9)),improved_fraction=float((sign*d>0).mean())))
write('paired_statistics.csv',paired)
assert all(sha(p)==h for p,h in hashes.items())
(out/'quality_complete.json').write_text(json.dumps(dict(completed=True,model_sha_unchanged=True,
    quality=summary,paired=paired),indent=2))
print('QUALITY_COMPLETE',flush=True)
