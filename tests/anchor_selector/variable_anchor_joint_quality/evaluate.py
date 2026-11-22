"""原 joint 可变成员质量测试；复用固定模型与已有 nominal 质量，只补充新 same-K 推理。"""
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

root = Path('/home/prmi/zwj/variable-anchor-joint-quality-20261021')
repo = Path('/home/prmi/zwj/variable-anchor-20261021/source')
out = root/'validation'
dataset = Path('/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz')
sys.path.insert(0, str(repo/'python'))
from evo.tools import settings
settings.SETTINGS.plot_backend = 'Agg'
import selector

if len(sys.argv) > 1 and sys.argv[1] == 'entry':
    policy = sys.argv.pop(2)
    sys.argv.pop(1)
    dest = Path(sys.argv[sys.argv.index('--output_path')+1])
    plan = json.loads((out/'evaluation_plan.json').read_text())
    original = selector.build_selector_mask
    @torch.no_grad()
    def mask(model, g, metadata, ratio, config, temperature=1., protection_enabled=True):
        selected, stats = original(model, g, metadata, ratio, config, temperature, protection_enabled)
        stats['raw_cached_selected'] = int(selected.sum())
        if policy == 'same_k':
            k = int(dest.name[1:])
            p = model(g.get_xyz, None, g.get_scaling, torch.tensor([[ratio]], device='cuda'), tau=temperature,
                      sh_dc=g.get_features[:,:1], opacity=g.get_opacity)
            p = selector.mask_inactive_slots(model, p, g.get_opacity)
            margin = (torch.logit(p.double())-torch.logit(metadata['cached_uniform'].double())).cpu().numpy()
            order = np.lexsort((np.arange(len(p)), -margin))
            ids = torch.as_tensor(order[:k].copy(), device='cuda')
            assert not np.isnan(margin).any() and (p[ids] > 0).all()
            selected = torch.zeros_like(selected)
            selected[ids] = 1
            stats.update(active_gaussians=k, actual_active=k, common_absolute_k=k)
        stats.update(selection_policy=policy, input_ratio=ratio)
        np.savez_compressed(dest/'selected_rows.npz', rows=torch.nonzero(selected).flatten().cpu().numpy())
        alpha = g.get_opacity.flatten()
        stats['selected_below_cutoff'] = int(((alpha < 1/255)&selected.bool()).sum())
        stats['selected_below_002'] = int(((alpha < .02)&selected.bool()).sum())
        return selected, stats
    selector.build_selector_mask = mask
    import renderer
    import cv2
    from diff_gaussian_rasterization import _C
    original_render = renderer.render_prepared
    photos = []
    images = np.loadtxt(dataset/'rgb.txt', dtype=str, comments='#')[:,1]
    def render(cam, data, bg, scaling_modifier=1., measure_time=False):
        result = original_render(cam, data, bg, scaling_modifier, measure_time)
        if measure_time:
            frame = plan['frames'][len(photos)]
            gt = cv2.cvtColor(cv2.imread(str(dataset/images[frame])), cv2.COLOR_BGR2RGB)
            gt = torch.as_tensor(gt, device='cuda').permute(2,0,1).float().contiguous()/255.
            prediction = result['render'].contiguous()
            photos.append(float(.8*(prediction-gt).abs().mean()+.2*(1-_C.fusedssim(.0001,.0009,prediction,gt).mean())))
        return result
    renderer.render_prepared = render
    runpy.run_path(str(repo/'python/run.py'), run_name='__main__')
    assert len(photos) == len(plan['frames'])
    np.savetxt(dest/'photo.txt', photos)
    raise SystemExit()

from evo.core import sync
from evo.tools import file_interface
from plyfile import PlyData
jobs = dict(fixed=Path('/home/prmi/zwj/anchor-slot-deactivation-20261021/enabled_seed2026'),
            variable=root/'variable_seed2026')
assert json.loads((out/'status.json').read_text())['stage'] == 'TRAINED'
assert json.loads((out/'protocol_amendment.json').read_text())['baseline'] == str(jobs['fixed'])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
hashes, population, covered = {}, {}, []
rgb = np.loadtxt(dataset/'rgb.txt', dtype=str, comments='#')[:,0].astype(float)
for name, run in jobs.items():
    state = json.loads((run/'10000_shutdown/ply/ultrafast_rotation_state.json').read_text())
    assert not state.get('serial_full_lod', False) and state.get('post_topology_full_steps', 0) == 0
    assert state['ultrafast_total_normal_steps'] == 10000
    gt = file_interface.read_tum_trajectory_file(str(dataset/'groundtruth.txt'))
    est = file_interface.read_tum_trajectory_file(str(run/'CameraTrajectory_TUM.txt'))
    _, trajectory = sync.associate_trajectories(gt, est, max_diff=.08)
    ids = [int(np.argmin(abs(rgb-t))) for t in trajectory.timestamps if abs(rgb-t).min() < .08]
    assert len(ids) == len(set(ids))
    covered.append(set(ids))
    cp = run/'10000_shutdown/ply'
    cloud = cp/'point_cloud/iteration_10000'
    population[name] = len(PlyData.read(cloud/'point_cloud.ply')['vertex'])
    for p in [cp/'selector.pt', cp/'transform_field.pt', cloud/'point_cloud.ply', cloud/'selector_metadata.ply']:
        hashes[str(p)] = sha(p)
frames = json.loads(Path('/home/prmi/zwj/anchor-slot-deactivation-20261021/validation/evaluation_plan.json').read_text())['frames']
assert len(frames) == 64 and set(frames) <= set.intersection(*covered)
lower, upper = int(np.ceil(.01*max(population.values()))), int(np.floor(.15*min(population.values())))
assert upper-lower >= 3, 'No shared budget interval inside trained ratio range'
budgets = np.rint(np.linspace(lower, upper, 4)).astype(int).tolist()
plan = dict(scene='fr1_xyz', seed=2026, frames=frames, population=population, model_sha256=hashes,
            common_k=budgets, lower=lower, upper=upper, ratio_policy='same-K inputs K/N per model; both within [0.01,0.15]',
            same_k='cached logit margin order, source row tie-break; diagnostic only, production still Cached Bernoulli',
            reuse_fixed_nominal='/home/prmi/zwj/serial-joint-lod-20261021/quality/baseline/cached_bernoulli',
            poses='each online model own estimated poses; same RGB frame IDs; asynchronous training trajectories differ',
            excluded_duplicate_fixed=str(root/'fixed_seed2026'), serial_joint=False)
(out/'evaluation_plan.json').write_text(json.dumps(plan, indent=2))
frames_path = out/'common_frames.json'
frames_path.write_text(json.dumps(dict(gt_indices=frames), indent=2))
old_root = Path('/home/prmi/zwj/serial-joint-lod-20261021')
old_plan = json.loads((old_root/'validation/evaluation_plan.json').read_text())
assert old_plan['frames'] == frames
assert all(old_plan['model_sha256'][p] == h for p, h in hashes.items() if p.startswith(str(jobs['fixed'])+'/'))
old_manifest = json.loads((old_root/'delivery_verification.json').read_text())['file_sha256']
summary, views, reuse = [], [], {}
metrics = ['psnr', 'ssim', 'lpips', 'photo']
env = dict(os.environ, OMP_NUM_THREADS='2', MPLBACKEND='Agg', QT_QPA_PLATFORM='offscreen')
def write(name, rows):
    with (out/name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
for name, run in jobs.items():
    cp = run/'10000_shutdown/ply'
    cases = [('cached_bernoulli', 'Full', None)]+[('cached_bernoulli', str(int(100*r)), r) for r in [.01,.05,.1,.15]]
    cases += [('same_k', 'K'+str(k), k/population[name]) for k in budgets]
    for policy, level, ratio in cases:
        dest = root/'quality'/name/policy/level
        dest.mkdir(parents=True, exist_ok=False)
        if name == 'fixed' and policy == 'cached_bernoulli':
            origin = old_root/'quality/baseline/cached_bernoulli'/level
            for p in origin.iterdir():
                if p.suffix in ['.txt', '.json', '.npz', '.log']:
                    assert sha(p) == old_manifest[str(p.relative_to(old_root))]
                    shutil.copy2(p, dest/p.name)
                    reuse[str((dest/p.name).relative_to(root))] = dict(source=str(p), sha256=sha(p))
        else:
            cmd = [sys.executable, __file__, 'entry', policy, str(run), str(dataset), '--output_path', str(dest),
                   '--evaluation-frame-set', str(frames_path), '--no-save-images']
            if ratio is not None:
                cmd += ['--selector', str(cp/'selector.pt'), '--selector-ratio', str(ratio), '--selector-config', str(cp/'selector_config.yaml')]
            with (dest/'evaluation.log').open('w') as log:
                subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        assert json.loads((dest/'evaluation_frames.json').read_text())['gt_indices'] == frames
        stats = json.loads((dest/'selector_stats.json').read_text())
        values = {m: np.loadtxt(dest/(m+'.txt')) for m in metrics}
        assert all(v.shape == (64,) and np.isfinite(v).all() for v in values.values())
        if policy == 'same_k':
            assert stats['active_gaussians'] == int(level[1:]) and .01-1e-9 <= ratio <= .15+1e-9
        row = dict(method=name, policy=policy, level=level, ratio=ratio, selected=stats['active_gaussians'],
                   total=stats['total_gaussians'], inactive=stats.get('inactive_slots',0),
                   selected_below_cutoff=stats.get('selected_below_cutoff'),
                   reused=(name == 'fixed' and policy == 'cached_bernoulli'), **{m:float(v.mean()) for m,v in values.items()})
        summary.append(row)
        for i, frame in enumerate(frames):
            views.append(dict(method=name, policy=policy, level=level, gt_index=frame, **{m:float(v[i]) for m,v in values.items()}))
        write('quality_summary.csv', summary)
        write('quality_per_view.csv', views)
        print('QUALITY', row, flush=True)
paired = []
for policy, levels in [('cached_bernoulli',['Full','1','5','10','15']), ('same_k',['K'+str(k) for k in budgets])]:
    for level in levels:
        for metric, sign in [('psnr',1),('ssim',1),('lpips',-1),('photo',-1)]:
            values = {name:np.array([r[metric] for r in views if (r['method'],r['policy'],r['level']) == (name,policy,level)]) for name in jobs}
            d = values['variable']-values['fixed']
            paired.append(dict(comparison='variable-fixed', policy=policy, level=level, metric=metric, mean=float(d.mean()),
                               median=float(np.median(d)), p10=float(np.quantile(d,.1)), p90=float(np.quantile(d,.9)), improved_fraction=float((sign*d>0).mean())))
write('paired_statistics.csv', paired)
assert all(sha(p) == h for p,h in hashes.items())
(out/'reused_fixed_evidence.json').write_text(json.dumps(reuse, indent=2))
(out/'quality_complete.json').write_text(json.dumps(dict(completed=True, model_sha_unchanged=True, arms=len(summary),
                                                       per_view_rows=len(views), paired_rows=len(paired)), indent=2))
print('QUALITY_COMPLETE', flush=True)
