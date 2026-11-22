"""原 checkpoint 只读诊断：逐 Anchor opacity、真实 alpha*T、移除低 opacity 点的敏感性。"""
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
from plyfile import PlyData

original=Path('/home/prmi/zwj/octree-explicit-20261021')
output=Path('/home/prmi/zwj/octree-explicit-diagnostics-20261021/frozen_evidence')
repo=original/'source'
sys.path.insert(0,str(repo/'python'))

def write_csv(path, rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def array(x): return x.detach().cpu().numpy()

if len(sys.argv)>1 and sys.argv[1]=='entry':
    seed=int(sys.argv[2]);mode=sys.argv[3]
    import renderer
    from evo.tools import settings
    settings.SETTINGS.plot_backend='Agg'
    full=renderer.render_prepared
    masses=None;visible=None;per_view=[];check_errors=[]
    def wrapper(cam, data, background, scaling_modifier=1.,measure_time=False):
        global masses,visible
        inputs=data['inputs']
        opacity=inputs['opacities'].reshape(-1)
        low=opacity<.02
        if mode=='drop_low':
            n=opacity.numel()
            subset={k:(v[~low].contiguous() if isinstance(v,torch.Tensor) and v.ndim and v.shape[0]==n else v)
                    for k,v in inputs.items()}
            return full(cam,dict(data,inputs=subset),background,scaling_modifier,measure_time)
        rendered=full(cam,data,background,scaling_modifier,measure_time)
        if measure_time:
            with torch.enable_grad():
                assert (background==0).all()
                colors=torch.ones((opacity.numel(),3),device='cuda',requires_grad=True)
                args=dict(inputs,dc=None,shs=None,colors_precomp=colors)
                rasterizer=renderer._make_rasterizer(cam,data['sh_degree'],background,scaling_modifier,inference_only=False)
                image,_=rasterizer(**args)
                grad=torch.autograd.grad(image[0].sum(),colors)[0][:,0]
                assert torch.isfinite(grad).all() and grad.min()>=-1e-5
                error=float(abs(grad.sum()-image[0].sum())/image[0].sum().clamp_min(1))
                assert error<2e-4,error
                check_errors.append(error)
                g=array(grad).astype(np.float64)
                if masses is None: masses=np.zeros_like(g);visible=np.zeros(len(g),np.int64)
                masses+=g;visible+=(g>1e-8)
                per_view.append(dict(index=len(per_view),total_alpha_T=float(g.sum()),
                    low_alpha_T=float(g[array(low)].sum()),low_mass_fraction=float(g[array(low)].sum()/g.sum()),
                    low_nonzero=int((g[array(low)]>1e-8).sum()),other_nonzero=int((g[~array(low)]>1e-8).sum())))
        return rendered
    renderer.render_prepared=wrapper
    run=original/f'explicit_seed{seed}'
    dest=output/f'seed{seed}'/mode
    sys.argv=[str(repo/'python/run.py'),str(run),'/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz',
              '--output_path',str(dest),'--evaluation-frame-set',str(output/'frames64.json'),'--no-save-images']
    runpy.run_path(str(repo/'python/run.py'),run_name='__main__')
    if mode=='full':
        np.savez_compressed(output/f'seed{seed}_contribution.npz',alpha_T_sum=masses,nonzero_view_count=visible)
        write_csv(output/f'seed{seed}_contribution_per_view.csv',per_view)
        (output/f'seed{seed}_contribution_correctness.json').write_text(json.dumps(dict(
            views=len(per_view),max_relative_conservation_error=max(check_errors),
            formula='sum_pixels alpha*T, measured by precomputed color VJP; no clipping, no training updates'),indent=2))
    raise SystemExit()

output.mkdir(parents=True,exist_ok=True)
plan=json.loads((original/'validation/evaluation_plan.json').read_text())
hashes={p:h for p,h in plan['model_sha256'].items() if '/explicit_seed' in p}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert all(sha(p)==h for p,h in hashes.items())
frames=json.loads((original/'validation/common_frames.json').read_text())['gt_indices']
frames=[frames[i] for i in np.linspace(0,len(frames)-1,64,dtype=int)]
(output/'frames64.json').write_text(json.dumps(dict(gt_indices=frames,policy='uniform, pre-quality, original common views'),indent=2))
stats=[];hist=[];all_rows=[]
for seed in (2026,2027):
    run=original/f'explicit_seed{seed}'
    for cloud in sorted(run.glob('*/ply/point_cloud/*/point_cloud.ply')):
        d=PlyData.read(cloud)['vertex'].data
        opacity=1/(1+np.exp(-np.clip(d['opacity'].astype(np.float64),-700,700)))
        m=len(d)//10;assert len(d)==10*m
        selector=torch.jit.load(str(cloud.parents[2]/'selector.pt'),map_location='cpu')
        b=dict(selector.named_buffers());uids=array(b['anchors.block_uids']);levels=array(b['anchors.levels']);keys=array(b['anchors.keys'])
        a=opacity.reshape(m,10);low=a<.02;counts=low.sum(1)
        stats.append(dict(seed=seed,checkpoint=cloud.parents[3].name,anchors=m,gaussians=len(d),
            low_points=int(low.sum()),low_point_fraction=float(low.mean()),
            anchors_with_low=int((counts>0).sum()),anchors_with_ge5_low=int((counts>=5).sum()),
            anchors_all10_low=int((counts==10).sum()),low_opacity_mean=float(a[low].mean()),
            low_opacity_median=float(np.median(a[low])),other_opacity_mean=float(a[~low].mean()),
            low_opacity_p90=float(np.quantile(a[low],.9)),low_opacity_max=float(a[low].max())))
        for level in [-1,0,1,2]:
            mask=np.ones(m,bool) if level==-1 else levels==level
            for n in range(11):hist.append(dict(seed=seed,checkpoint=cloud.parents[3].name,level=level,failed_members=n,anchors=int((counts[mask]==n).sum())))
        if cloud.parents[3].name=='10000_shutdown':
            for j in range(m):
                row=dict(seed=seed,uid=int(uids[j]),level=int(levels[j]),key_x=int(keys[j,0]),key_y=int(keys[j,1]),key_z=int(keys[j,2]),
                         failed_members=int(counts[j]),opacity_min=float(a[j].min()),opacity_mean=float(a[j].mean()),opacity_max=float(a[j].max()))
                row.update({f'opacity_slot{k}':float(a[j,k]) for k in range(10)})
                all_rows.append(row)
write_csv(output/'original_opacity_summary.csv',stats)
write_csv(output/'original_failed_member_histogram.csv',hist)
write_csv(output/'original_surviving_anchors.csv',all_rows)
env=dict(os.environ,OMP_NUM_THREADS='2',MPLBACKEND='Agg',QT_QPA_PLATFORM='offscreen')
for seed in (2026,2027):
    for mode in ('full','drop_low'):
        dest=output/f'seed{seed}'/mode;dest.mkdir(parents=True,exist_ok=False)
        with (dest/'evaluation.log').open('w') as log:
            subprocess.run([sys.executable,__file__,'entry',str(seed),mode],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
contributions=[];quality=[];gaussian_rows=[]
for seed in (2026,2027):
    records=[r for r in all_rows if r['seed']==seed]
    a=np.array([[r[f'opacity_slot{k}'] for k in range(10)] for r in records]).reshape(-1)
    low=a<.02
    data=np.load(output/f'seed{seed}_contribution.npz');mass=data['alpha_T_sum'];vis=data['nonzero_view_count']
    assert len(mass)==len(a)
    for name,mask in [('all',np.ones(len(a),bool)),('low_opacity',low),('other',~low)]:
        x=mass[mask];contributions.append(dict(seed=seed,group=name,points=int(mask.sum()),alpha_T_sum=float(x.sum()),
            total_mass_fraction=float(x.sum()/mass.sum()),mean_per_point=float(x.mean()),
            median_per_point=float(np.median(x)),p90_per_point=float(np.quantile(x,.9)),
            zero_contribution_points=int((x<=1e-8).sum()),zero_contribution_fraction=float((x<=1e-8).mean()),
            mean_opacity=float(a[mask].mean())))
    for i in range(len(a)):
        gaussian_rows.append(dict(seed=seed,row=i,uid=records[i//10]['uid'],slot=i%10,opacity=float(a[i]),
            would_prune_individually=bool(low[i]),alpha_T_sum=float(mass[i]),nonzero_views=int(vis[i])))
    for mode in ('full','drop_low'):
        dest=output/f'seed{seed}'/mode
        row=dict(seed=seed,mode=mode,views=64,render_gaussians=int((~low).sum()) if mode=='drop_low' else len(a))
        for metric in ['psnr','ssim','lpips']:row[metric]=float(np.loadtxt(dest/(metric+'.txt')).mean())
        quality.append(row)
write_csv(output/'contribution_summary.csv',contributions)
write_csv(output/'gaussian_contributions.csv',gaussian_rows)
write_csv(output/'low_opacity_ablation_quality.csv',quality)
assert all(sha(p)==h for p,h in hashes.items())
(output/'correctness.json').write_text(json.dumps(dict(original_models_unchanged=True,
    opacity_threshold=.02,prune_big_point_disabled=True,views=64,seeds=[2026,2027],
    posthoc_only=True,original_history_unavailable=['birth_observation_count','birth_key_history','death_mapping_step']),indent=2))
print(json.dumps(dict(opacity=stats,contribution=contributions,quality=quality),indent=2))
