"""冻结最终模型、64 个既定视图；生产 Cached Bernoulli 和 same-K 分开报告。"""
import csv
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import numpy as np
import torch

root=Path('/home/prmi/zwj/shared-selector-appearance-20261021');repo=root/'source';out=root/'validation'
dataset=Path('/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz')
sys.path.insert(0,str(repo/'python'))
from evo.tools import settings
settings.SETTINGS.plot_backend='Agg'
import selector

if len(sys.argv)>1 and sys.argv[1]=='entry':
    policy=sys.argv.pop(2);sys.argv.pop(1)
    dest=Path(sys.argv[sys.argv.index('--output_path')+1])
    plan=json.loads((out/'evaluation_plan.json').read_text())
    original=selector.build_selector_mask
    @torch.no_grad()
    def mask(model,g,metadata,ratio,config,temperature=1.,protection_enabled=True,**kwargs):
        selected,stats=original(model,g,metadata,ratio,config,temperature,protection_enabled,**kwargs)
        stats['raw_cached_selected']=int(selected.sum())
        if policy=='same_k':
            k=int(dest.name[1:])
            p=kwargs['forward_features'][0] if kwargs.get('forward_features') is not None else model(
                g.get_xyz,None,g.get_scaling,torch.tensor([[ratio]],device='cuda'),tau=temperature,
                sh_dc=g.get_features[:,:1],opacity=g.get_opacity)
            p=selector.mask_inactive_slots(model,p,g.get_opacity)
            margin=(torch.logit(p.double())-torch.logit(metadata['cached_uniform'].double())).cpu().numpy()
            order=np.lexsort((np.arange(len(p)),-margin));ids=torch.as_tensor(order[:k].copy(),device='cuda')
            assert not np.isnan(margin).any() and (p[ids]>0).all()
            selected=torch.zeros_like(selected);selected[ids]=1
            stats.update(active_gaussians=k,actual_active=k,common_absolute_k=k)
        stats.update(selection_policy=policy,input_ratio=ratio)
        np.savez_compressed(dest/'selected_rows.npz',rows=torch.nonzero(selected).flatten().cpu().numpy())
        return selected,stats
    selector.build_selector_mask=mask
    if dest.parts[-3]=='appearance_off':
        original_compensation=selector.compensated_attributes
        def geometry_only(*args,**kwargs):
            result=original_compensation(*args,**kwargs)
            if result is not None:
                result.pop('opacity_logits',None);result.pop('features_dc',None)
            return result
        selector.compensated_attributes=geometry_only
    import renderer
    import cv2
    from diff_gaussian_rasterization import _C
    original_render=renderer.render_prepared
    photos=[];images=np.loadtxt(dataset/'rgb.txt',dtype=str,comments='#')[:,1]
    def render(cam,data,bg,scaling_modifier=1.,measure_time=False):
        result=original_render(cam,data,bg,scaling_modifier,measure_time)
        if measure_time:
            frame=plan['frames'][len(photos)]
            gt=cv2.cvtColor(cv2.imread(str(dataset/images[frame])),cv2.COLOR_BGR2RGB)
            gt=torch.as_tensor(gt,device='cuda').permute(2,0,1).float().contiguous()/255.
            prediction=result['render'].contiguous()
            photos.append(float(.8*(prediction-gt).abs().mean()+.2*(1-_C.fusedssim(.0001,.0009,prediction,gt).mean())))
        return result
    renderer.render_prepared=render
    runpy.run_path(str(repo/'python/run.py'),run_name='__main__')
    assert len(photos)==len(plan['frames'])
    np.savetxt(dest/'photo.txt',photos)
    raise SystemExit()

from evo.core import sync
from evo.tools import file_interface
from plyfile import PlyData
assert json.loads((out/'training_status.json').read_text())['stage']=='TRAINED'
baseline=Path('/home/prmi/zwj/shared-selector-compensation-20261021')
jobs=dict(geometry_v1=baseline/'candidate_seed2026',appearance_v2=root/'candidate_seed2026')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
hashes={};population={};covered=[]
rgb=np.loadtxt(dataset/'rgb.txt',dtype=str,comments='#')[:,0].astype(float)
for name,run in jobs.items():
    cp=run/'10000_shutdown/ply';cloud=cp/'point_cloud/iteration_10000'
    population[name]=len(PlyData.read(cloud/'point_cloud.ply')['vertex'])
    for p in [cp/'selector.pt',cp/'selector_config.yaml',cloud/'point_cloud.ply',cloud/'selector_metadata.ply']:
        hashes[str(p)]=sha(p)
    if (cp/'transform_field.pt').exists():hashes[str(cp/'transform_field.pt')]=sha(cp/'transform_field.pt')
    gt=file_interface.read_tum_trajectory_file(str(dataset/'groundtruth.txt'))
    est=file_interface.read_tum_trajectory_file(str(run/'CameraTrajectory_TUM.txt'))
    _,trajectory=sync.associate_trajectories(gt,est,max_diff=.08)
    covered.append({int(np.argmin(abs(rgb-t))) for t in trajectory.timestamps if abs(rgb-t).min()<.08})
frames=json.loads((out/'training_protocol.json').read_text())['frames']
assert len(frames)==64 and set(frames)<=set.intersection(*covered),'Locked evaluation view missing'
lower=int(np.ceil(.01*max(population.values())));upper=int(np.floor(.15*min(population.values())))
assert upper-lower>=3
budgets=np.rint(np.linspace(lower,upper,4)).astype(int).tolist()
plan=dict(scene='fr1_xyz',seed=2026,frames=frames,population=population,model_sha256=hashes,common_k=budgets,
    input_ratio='same-K uses K/N for each model, inside [0.01,0.15]',
    same_k='cached logit margin exact-K diagnostic only; production Cached Bernoulli unchanged',
    diagnostic='appearance_off uses identical candidate checkpoint, selected rows and ratio',
    poses='each online model own estimated poses; locked same RGB IDs; asynchronous training trajectories differ')
(out/'evaluation_plan.json').write_text(json.dumps(plan,indent=2))
(out/'common_frames.json').write_text(json.dumps(dict(gt_indices=frames)))
metrics=['psnr','ssim','lpips','photo'];summary=[];views=[];reused={}
env=dict(os.environ,OMP_NUM_THREADS='2',MPLBACKEND='Agg',QT_QPA_PLATFORM='offscreen')
def write(name,rows):
    with (out/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
for name in ['geometry_v1','appearance_v2','appearance_off']:
    run=jobs['appearance_v2' if name=='appearance_off' else name];cp=run/'10000_shutdown/ply'
    cases=[('cached_bernoulli',str(int(100*r)),r) for r in [.01,.05,.1,.15]]
    if name!='appearance_off':
        cases=[('cached_bernoulli','Full',None)]+cases+[('same_k','K'+str(k),k/population[name]) for k in budgets]
    for policy,level,ratio in cases:
        dest=root/'quality'/name/policy/level;dest.mkdir(parents=True,exist_ok=False)
        if name=='geometry_v1' and policy=='cached_bernoulli':
            origin=baseline/'quality/shared_compensation/cached_bernoulli'/level
            for p in origin.iterdir():
                if p.suffix in ['.txt','.json','.npz','.log']:
                    shutil.copy2(p,dest/p.name);reused[str(dest/p.name)]=dict(source=str(p),sha256=sha(p))
        else:
            cmd=[sys.executable,__file__,'entry',policy,str(run),str(dataset),'--output_path',str(dest),
                 '--evaluation-frame-set',str(out/'common_frames.json'),'--no-save-images']
            if ratio is not None:cmd+=['--selector',str(cp/'selector.pt'),'--selector-ratio',str(ratio),'--selector-config',str(cp/'selector_config.yaml')]
            with (dest/'evaluation.log').open('w') as log:
                subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert json.loads((dest/'evaluation_frames.json').read_text())['gt_indices']==frames
        stats=json.loads((dest/'selector_stats.json').read_text())
        values={m:np.loadtxt(dest/(m+'.txt')) for m in metrics}
        assert all(v.shape==(64,) and np.isfinite(v).all() for v in values.values())
        if policy=='same_k':assert stats['active_gaussians']==int(level[1:]) and .01<=ratio<=.15
        summary.append(dict(method=name,policy=policy,level=level,ratio=ratio,selected=stats['active_gaussians'],
                            total=stats['total_gaussians'],compensated=stats.get('selector_compensation_applied',False),appearance=stats.get('appearance_compensation_applied',False),
                            **{m:float(v.mean()) for m,v in values.items()}))
        for i,frame in enumerate(frames):views.append(dict(method=name,policy=policy,level=level,gt_index=frame,**{m:float(v[i]) for m,v in values.items()}))
        write('quality_summary.csv',summary);write('quality_per_view.csv',views)
        print('QUALITY',summary[-1],flush=True)
for level in ['1','5','10','15']:
    load=lambda name:np.load(root/'quality'/name/'cached_bernoulli'/level/'selected_rows.npz')['rows']
    assert np.array_equal(load('appearance_v2'),load('appearance_off'))
paired=[]
for right,left in [('appearance_v2','geometry_v1'),('appearance_v2','appearance_off')]:
    cases=[('cached_bernoulli',x) for x in (['Full','1','5','10','15'] if left=='geometry_v1' else ['1','5','10','15'])]
    if left=='geometry_v1':cases+=[('same_k','K'+str(k)) for k in budgets]
    for policy,level in cases:
        for metric,sign in [('psnr',1),('ssim',1),('lpips',-1),('photo',-1)]:
            take=lambda n:np.array([r[metric] for r in views if (r['method'],r['policy'],r['level'])==(n,policy,level)])
            d=take(right)-take(left)
            paired.append(dict(comparison=right+'-'+left,policy=policy,level=level,metric=metric,mean=float(d.mean()),
                median=float(np.median(d)),p10=float(np.quantile(d,.1)),p90=float(np.quantile(d,.9)),improved_fraction=float((sign*d>0).mean())))
write('paired_statistics.csv',paired)
assert all(sha(p)==h for p,h in hashes.items())
(out/'reused_baseline.json').write_text(json.dumps(reused,indent=2))
(out/'quality_complete.json').write_text(json.dumps(dict(completed=True,arms=len(summary),per_view_rows=len(views),paired_rows=len(paired),model_sha_unchanged=True),indent=2))
print('QUALITY_COMPLETE',flush=True)
