"""只评价新联合模型；局部预算匹配全局逐帧实际Alpha贡献数，不进行训练。"""
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
OUT = ROOT/'evaluation/matched_visible_joint_20261105'
sys.path[:0] = [str(ROOT/'evaluation/python'), str(ROOT/'evaluation/python/diff-gaussian-rasterization')]
import numpy as np
import torch
from budget_viewer import BudgetScene
from renderer import render_prepared
from utils import MiniCam, focal2fov

RATIOS = [.1, .2, .3, .4, .5]
FIELDS = ['method', 'source_ratio', 'network_ratio', 'frame', 'timestamp', 'image', 'bank',
          'N', 'candidate_N', 'eligible_N', 'requested_K', 'K', 'projected_K', 'contributing_K',
          'budget_shortfall', 'global_reference_K', 'global_reference_projected_K',
          'global_reference_contributing_K', 'global_psnr', 'psnr', 'delta_psnr_to_global',
          'photo_l1', 'selection_sha256', 'metrics_source']


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_counted(binding, camera, data, background):
    inputs = data['inputs']; empty = torch.Tensor([])
    value = lambda name: inputs[name] if inputs[name] is not None else empty
    return binding.render_with_counts(background, value('means3D'), value('colors_precomp'), value('opacities'),
        value('scales'), value('rotations'), 1., value('cov3D_precomp'), camera.world_view_transform,
        camera.full_proj_transform, math.tan(camera.FoVx*.5), math.tan(camera.FoVy*.5),
        camera.image_height, camera.image_width, value('dc'), value('shs'), data['sh_degree'],
        camera.camera_center, False, False, True, binding.Mode.Inference)


@torch.no_grad()
def main():
    OUT.mkdir(exist_ok=True); (OUT/'exports').mkdir(exist_ok=True)
    assert not (OUT/'PER_VIEW.csv').exists(), '已有评价文件，禁止覆盖'
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    started = time.monotonic()
    previous = ROOT/'evaluation/reports'
    assert json.loads((ROOT/'evaluation/STATUS.json').read_text())['state'] == 'complete'
    checkpoint = ROOT/'runs/full_fine_joint/23560_shutdown/ply'
    protected = {p: h for p, h in json.loads((previous/'INPUT_INTEGRITY.json').read_text())['after'].items()
                 if '/full_fine_joint/' in p}
    binding_path = Path('/home/prmi/zwj/experiments/fr1xyz_viewer_budget_comparison_20261105/build/fr1xyz_production_counts.so')
    for p in [previous/'PER_VIEW.csv', binding_path, Path(__file__)] + list((ROOT/'evaluation/python').rglob('*.py')):
        protected[str(p)] = sha(p)
    assert protected and all(sha(Path(p)) == h for p, h in protected.items()), '旧评价输入身份已变化'
    save(OUT/'INPUTS_BEFORE.json', protected)
    spec = importlib.util.spec_from_file_location('fr1xyz_production_counts', binding_path)
    binding = importlib.util.module_from_spec(spec); spec.loader.exec_module(binding)
    source = [r for r in csv.DictReader((previous/'PER_VIEW.csv').open()) if r['group'] == 'joint']
    by = {(r['scope'], float(r['ratio']) if r['ratio'] else None, int(r['frame'])): r for r in source}
    scene = BudgetScene(checkpoint, '/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz', 5000, 0, None, 'pytorch')
    assert len(scene.views) == 792 and scene.renderer.total == 13298
    exports = []
    for ratio in RATIOS:
        data, ids, stats = scene.renderer.prepare(None, global_ratio=ratio)
        path = OUT/'exports'/f'global_ratio_{ratio:g}.pt'
        torch.save({'source_checkpoint': str(checkpoint), 'ratio': ratio, 'ids': ids.cpu(), 'stats': stats,
                    'sh_degree': data['sh_degree'], 'inputs': {k: v.cpu() if isinstance(v, torch.Tensor) else v
                                                             for k, v in data['inputs'].items()}}, path)
        exports.append({'ratio': ratio, 'K': len(ids), 'path': str(path), 'sha256': sha(path)})
    save(OUT/'GLOBAL_EXPORTS.json', exports)
    checks, rows = [], []
    with (OUT/'PER_VIEW.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader()
        for frame, view in enumerate(scene.views):
            c = scene.calibration
            camera = MiniCam(c['width'], c['height'], focal2fov(c['fx'], c['width']),
                            focal2fov(c['fy'], c['height']), np.linalg.inv(np.asarray(view['pose'])))
            truth = scene.ground_truth(frame)
            assert truth is not None
            for ratio in [None]+RATIOS:
                reference = by[('full' if ratio is None else 'global', ratio, frame)]
                assert reference['image'] == view['image'] and float(reference['timestamp']) == view['timestamp']
                methods = ['full'] if ratio is None else ['global', 'view_ratio', 'view_matched']
                for method in methods:
                    if method != 'view_matched':
                        scope = 'view' if method == 'view_ratio' else method
                        old = by[(scope, ratio, frame)]
                        row = {name: old[name] for name in ['frame', 'timestamp', 'image', 'bank', 'N', 'candidate_N',
                            'eligible_N', 'requested_K', 'K', 'projected_K', 'contributing_K', 'budget_shortfall',
                            'selection_sha256', 'psnr', 'photo_l1']}
                        row.update(method=method, source_ratio=ratio, network_ratio=ratio, metrics_source='reused_frozen_evaluation')
                    else:
                        budget = int(reference['contributing_K'])
                        data, ids, stats = scene.renderer.prepare(camera, budget=budget)
                        assert len(ids) == min(budget, stats['eligible_candidates'])
                        expected = min(1., budget/stats['candidate_gaussians']) if budget and stats['candidate_gaussians'] else None
                        assert stats['ratio'] == expected
                        output = render_counted(binding, camera, data, scene.background)
                        image = output[2].clamp(0, 1)
                        difference = image-truth
                        quality = float(-10*torch.log10(difference.square().mean()))
                        assert torch.isfinite(image).all() and math.isfinite(quality)
                        projected, contributed = int((output[3] > 0).sum()), int((output[5] > 0).sum())
                        assert 0 <= contributed <= projected <= len(ids)
                        if frame in [0, 396, 791]:
                            normal = render_prepared(camera, data, scene.background, return_radii=True)
                            assert torch.equal(normal['render'].view(torch.int32), output[2].view(torch.int32))
                            assert torch.equal(normal['radii'], output[3])
                            checks.append({'method': method, 'ratio': ratio, 'frame': frame, 'bitwise_render_parity': True})
                        row = dict(method=method, source_ratio=ratio, network_ratio=stats['ratio'], frame=frame,
                            timestamp=view['timestamp'], image=view['image'], bank=reference['bank'], N=scene.renderer.total,
                            candidate_N=stats['candidate_gaussians'], eligible_N=stats['eligible_candidates'],
                            requested_K=budget, K=len(ids), projected_K=projected, contributing_K=contributed,
                            budget_shortfall=stats['budget_shortfall'], psnr=quality, photo_l1=float(difference.abs().mean()),
                            selection_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(), metrics_source='new_matched_inference')
                        del output, data, image
                    row.update(global_reference_K=reference['K'], global_reference_projected_K=reference['projected_K'],
                               global_reference_contributing_K=reference['contributing_K'], global_psnr=reference['psnr'],
                               delta_psnr_to_global=float(row['psnr'])-float(reference['psnr']))
                    writer.writerow(row); rows.append(row)
                    if method in ['global', 'view_ratio'] and frame in [0, 396, 791]:
                        data, ids, _ = scene.renderer.prepare(None, global_ratio=ratio) if method == 'global' else scene.renderer.prepare(camera, ratio_budget=ratio)
                        check = render_counted(binding, camera, data, scene.background)
                        assert hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest() == row['selection_sha256']
                        assert int((check[3] > 0).sum()) == int(row['projected_K'])
                        assert int((check[5] > 0).sum()) == int(row['contributing_K'])
                        error = abs(float(-10*torch.log10((check[2].clamp(0, 1)-truth).square().mean()))-float(row['psnr']))
                        assert error < 1e-4, error
                        checks.append({'method': method, 'ratio': ratio, 'frame': frame, 'selection_and_counts_reproduced': True, 'psnr_abs_error': error})
                        del check, data
            if frame % 25 == 0 or frame == 791:
                stream.flush()
                state = dict(state='running', frame=frame+1, frames=792, rows=len(rows), seconds=time.monotonic()-started)
                save(OUT/'STATUS.json', state); print(json.dumps(state), flush=True)
    after = {p: sha(Path(p)) for p in protected}
    save(OUT/'INPUTS_AFTER.json', after); assert protected == after
    assert len(rows) == 792*16
    save(OUT/'VALIDATION.json', checks)
    save(OUT/'STATUS.json', dict(state='complete', model='joint', frames=792, rows=len(rows), new_matched_renders=3960,
        reused_rows=8712, verification_renders=45, images_saved=0, training_steps=0, inputs_unchanged=True,
        seconds=time.monotonic()-started))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        OUT.mkdir(exist_ok=True)
        save(OUT/'STATUS.json', {'state': 'failed', 'error': repr(error)})
        raise
