"""新联合模型的单视角固定点数预算评价；复用已验收渲染及计数路径。"""
import csv
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path

import evaluate_matched_joint as base

ROOT = base.ROOT
OUT = ROOT/'evaluation/fixed_budget_joint_20261105'
BUDGETS = [1000, 2000, 3000, 4000, 5000, 10000]


@base.torch.no_grad()
def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'PER_VIEW.csv').exists(), '禁止覆盖已有评价'
    base.torch.set_num_threads(2)
    base.torch.backends.cuda.matmul.allow_tf32 = False
    base.torch.backends.cudnn.allow_tf32 = False
    base.torch.backends.cudnn.benchmark = False
    started = time.monotonic()
    previous = ROOT/'evaluation/reports'
    checkpoint = ROOT/'runs/full_fine_joint/23560_shutdown/ply'
    protected = {p: h for p, h in json.loads((previous/'INPUT_INTEGRITY.json').read_text())['after'].items()
                 if '/full_fine_joint/' in p}
    binding_path = Path('/home/prmi/zwj/experiments/fr1xyz_viewer_budget_comparison_20261105/build/fr1xyz_production_counts.so')
    for path in [Path(__file__), Path(base.__file__), binding_path]+list((ROOT/'evaluation/python').rglob('*.py')):
        protected[str(path)] = base.sha(path)
    assert all(base.sha(Path(p)) == h for p, h in protected.items())
    base.save(OUT/'INPUTS_BEFORE.json', protected)
    spec = importlib.util.spec_from_file_location('fr1xyz_production_counts', binding_path)
    binding = importlib.util.module_from_spec(spec); spec.loader.exec_module(binding)
    scene = base.BudgetScene(checkpoint, '/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz', 5000, 0, None, 'pytorch')
    assert len(scene.views) == 792 and scene.renderer.total == 13298
    full = {int(r['frame']): r for r in csv.DictReader((previous/'PER_VIEW.csv').open())
            if r['group'] == 'joint' and r['scope'] == 'full'}
    checks, rows = [], []
    with (OUT/'PER_VIEW.csv').open('w', newline='') as stream:
        writer = None
        for frame, view in enumerate(scene.views):
            assert full[frame]['image'] == view['image']
            c = scene.calibration
            camera = base.MiniCam(c['width'], c['height'], base.focal2fov(c['fx'], c['width']),
                base.focal2fov(c['fy'], c['height']), base.np.linalg.inv(base.np.asarray(view['pose'])))
            truth = scene.ground_truth(frame)
            assert truth is not None
            for budget in BUDGETS:
                data, ids, stats = scene.renderer.prepare(camera, budget=budget)
                assert len(ids) == min(budget, stats['eligible_candidates'])
                assert stats['ratio'] == min(1., budget/stats['candidate_gaussians'])
                output = base.render_counted(binding, camera, data, scene.background)
                image = output[2].clamp(0, 1)
                difference = image-truth
                quality = float(-10*base.torch.log10(difference.square().mean()))
                assert base.torch.isfinite(image).all() and math.isfinite(quality)
                projected, contributed = int((output[3] > 0).sum()), int((output[5] > 0).sum())
                assert 0 <= contributed <= projected <= len(ids)
                if frame in [0, 396, 791]:
                    normal = base.render_prepared(camera, data, scene.background, return_radii=True)
                    assert base.torch.equal(normal['render'].view(base.torch.int32), output[2].view(base.torch.int32))
                    assert base.torch.equal(normal['radii'], output[3])
                    checks.append({'frame': frame, 'budget': budget, 'color_radii_bitwise': True})
                row = dict(budget=budget, frame=frame, timestamp=view['timestamp'], image=view['image'],
                    bank=full[frame]['bank'], N=scene.renderer.total, visible_anchors=stats['visible_anchors'],
                    candidate_N=stats['candidate_gaussians'], eligible_N=stats['eligible_candidates'],
                    K=len(ids), projected_K=projected, contributing_K=contributed,
                    budget_shortfall=stats['budget_shortfall'], network_ratio=stats['ratio'],
                    ratio_in_training_range=stats['ratio_in_training_range'], psnr=quality,
                    photo_l1=float(difference.abs().mean()), full_psnr=float(full[frame]['psnr']),
                    selection_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest())
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader()
                writer.writerow(row); rows.append(row)
                del output, data, image
            if frame % 25 == 0 or frame == 791:
                stream.flush()
                state = dict(state='running', frame=frame+1, frames=792, rows=len(rows), seconds=time.monotonic()-started)
                base.save(OUT/'STATUS.json', state); print(json.dumps(state), flush=True)
    after = {p: base.sha(Path(p)) for p in protected}
    base.save(OUT/'INPUTS_AFTER.json', after); assert protected == after
    assert len(rows) == 792*6
    base.save(OUT/'VALIDATION.json', checks)
    base.save(OUT/'STATUS.json', dict(state='complete', model='joint', frames=792, budgets=BUDGETS,
        rows=len(rows), renders=4752, parity_checks=len(checks), images_saved=0, training_steps=0,
        inputs_unchanged=True, seconds=time.monotonic()-started))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        OUT.mkdir(exist_ok=True)
        base.save(OUT/'STATUS.json', {'state': 'failed', 'error': repr(error)})
        raise
