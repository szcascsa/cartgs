"""冻结三组最终模型，数值评价整条RGB轨迹；不保存渲染图片。"""
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'evaluation/python'), str(ROOT/'evaluation/python/diff-gaussian-rasterization')]
import numpy as np
import torch
from budget_viewer import BudgetScene
from renderer import render_prepared
from utils import MiniCam, focal2fov
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

RATIOS = [.01, .03, .05, .075, .1, .2, .3, .4, .5, .6, .7]
TRAIN_RATIOS = [.01, .05, .1, .3, .5, .7]
CASES = [('full', 'full', None)] + [(f'{scope}_{ratio:g}', scope, ratio)
    for scope in ['global', 'view'] for ratio in RATIOS]
FIELDS = ['group', 'case', 'scope', 'ratio', 'ratio_split', 'frame', 'timestamp', 'image', 'bank',
          'N', 'candidate_N', 'eligible_N', 'requested_K', 'K', 'projected_K', 'contributing_K',
          'budget_shortfall', 'selection_sha256', 'psnr', 'ssim', 'lpips', 'photo_l1']


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counted_render(binding, camera, data, background):
    inputs = data['inputs']; empty = torch.Tensor([])
    value = lambda key: inputs[key] if inputs[key] is not None else empty
    return binding.render_with_counts(background, value('means3D'), value('colors_precomp'), value('opacities'),
        value('scales'), value('rotations'), 1., value('cov3D_precomp'), camera.world_view_transform,
        camera.full_proj_transform, math.tan(camera.FoVx*.5), math.tan(camera.FoVy*.5),
        camera.image_height, camera.image_width, value('dc'), value('shs'), data['sh_degree'],
        camera.camera_center, False, False, True, binding.Mode.Inference)


@torch.no_grad()
def main():
    assert json.loads((ROOT/'reports/FULL_EXPERIMENT_STATUS.json').read_text())['state'] == 'training_complete'
    destination = ROOT/'evaluation/reports'; destination.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    binding_path = Path('/home/prmi/zwj/experiments/fr1xyz_viewer_budget_comparison_20261105/build/fr1xyz_production_counts.so')
    spec = importlib.util.spec_from_file_location('fr1xyz_production_counts', binding_path)
    binding = importlib.util.module_from_spec(spec); spec.loader.exec_module(binding)
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True, reduction='none').cuda()
    started = time.monotonic(); initial_hashes = {}; frame_keys = None; parity = []
    all_rows = []; geometry = []
    with (destination/'PER_VIEW.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader()
        for group in ['baseline', 'joint', 'global_control']:
            folder = ROOT/'runs'/('full_fine_'+group); checkpoint = folder/'23560_shutdown/ply'
            assert checkpoint.is_dir()
            for path in [checkpoint/'selector.pt', checkpoint/'selector_config.yaml', checkpoint/'cameras.json',
                         checkpoint/'point_cloud/iteration_23560/point_cloud.ply', folder/'CameraTrajectory_TUM.txt']:
                initial_hashes[str(path)] = sha(path)
            scene = BudgetScene(checkpoint, '/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz',
                                5000, 0, None, 'pytorch')
            assert len(scene.views) == 792 and scene.renderer.total == 13298
            keys = [(view['timestamp'], view['image']) for view in scene.views]
            if frame_keys is None:
                frame_keys = keys
                save(destination/'FRAME_ORDER.json', keys)
            assert keys == frame_keys, 'Three groups must share all RGB frames and camera trajectory'
            cameras = json.loads((checkpoint/'cameras.json').read_text())
            dataset = Path('/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz')
            bank = {str((dataset/c['img_name']).resolve()) for c in cameras}
            calibration = scene.calibration
            images, truths, pending = [], [], []

            def flush():
                if not pending:
                    return
                predictions, targets = torch.stack(images), torch.stack(truths)
                difference = predictions-targets
                mse = difference.square().mean((1, 2, 3))
                quality = -10*torch.log10(mse)
                structure = structural_similarity_index_measure(predictions, targets, data_range=1., reduction='none')
                perceptual = lpips(predictions, targets); lpips.reset()
                photo = difference.abs().mean((1, 2, 3))
                metrics = torch.stack([quality, structure, perceptual.reshape(-1), photo], 1)
                assert torch.isfinite(metrics).all()
                for row, values in zip(pending, metrics.cpu().tolist()):
                    row.update(zip(['psnr', 'ssim', 'lpips', 'photo_l1'], values))
                    writer.writerow(row); all_rows.append(row)
                images.clear(); truths.clear(); pending.clear()

            for frame, view in enumerate(scene.views):
                camera = MiniCam(calibration['width'], calibration['height'], focal2fov(calibration['fx'], calibration['width']),
                    focal2fov(calibration['fy'], calibration['height']), np.linalg.inv(np.asarray(view['pose'])))
                truth = scene.ground_truth(frame)
                assert truth is not None
                for name, scope, ratio in CASES:
                    if scope == 'full':
                        data, ids = scene.full_data, scene.full_ids
                        stats = dict(candidate_gaussians=scene.renderer.total, eligible_candidates=scene.renderer.total,
                                     requested_budget=scene.renderer.total, budget_shortfall=0)
                    elif scope == 'global':
                        data, ids, stats = scene.renderer.prepare(None, global_ratio=ratio)
                    else:
                        data, ids, stats = scene.renderer.prepare(camera, ratio_budget=ratio)
                        assert len(ids) == min(math.floor(ratio*stats['candidate_gaussians']), stats['eligible_candidates'])
                    rendered = counted_render(binding, camera, data, scene.background)
                    image = rendered[2].clamp(0, 1)
                    selected = int(ids.numel()); projected = int((rendered[3] > 0).sum())
                    contributing = int((rendered[5] > 0).sum())
                    assert 0 <= contributing <= projected <= selected
                    if frame in [0, len(scene.views)//2, len(scene.views)-1]:
                        normal = render_prepared(camera, data, scene.background, return_radii=True)
                        assert torch.equal(normal['render'].view(torch.int32), rendered[2].view(torch.int32))
                        assert torch.equal(normal['radii'], rendered[3])
                        parity.append({'group': group, 'frame': frame, 'case': name, 'bitwise': True})
                        if scope != 'full' and selected:
                            inputs = data['inputs']; pc = scene.gaussians; anchors = scene.model.anchors
                            owners = anchors.rows[ids]
                            movement = (inputs['means3D']-pc.get_xyz[ids]).norm(dim=1)/anchors.offset_scales[owners].reshape(-1)
                            scale_factor = (inputs['scales']/pc.get_scaling[ids]).amax(1)
                            rotation = 2*torch.acos((inputs['rotations']*pc.get_rotation[ids]).sum(1).abs().clamp(0, 1))
                            for metric, values in [('xyz_anchor_units', movement), ('scale_factor', scale_factor), ('rotation_radians', rotation)]:
                                quantiles = torch.quantile(values, torch.tensor([.5, .9, .99], device=values.device)).cpu().tolist()
                                geometry.append({'group': group, 'frame': frame, 'scope': scope, 'ratio': ratio, 'metric': metric,
                                                 'K': selected, 'p50': quantiles[0], 'p90': quantiles[1], 'p99': quantiles[2], 'max': values.max().item()})
                    row = dict(group=group, case=name, scope=scope, ratio=ratio,
                        ratio_split='full' if ratio is None else 'train' if ratio in TRAIN_RATIOS else 'unseen',
                        frame=frame, timestamp=view['timestamp'], image=view['image'], bank=str(Path(view['image']).resolve()) in bank,
                        N=scene.renderer.total, candidate_N=stats['candidate_gaussians'], eligible_N=stats['eligible_candidates'],
                        requested_K=stats['requested_budget'], K=selected, projected_K=projected, contributing_K=contributing,
                        budget_shortfall=stats['budget_shortfall'], selection_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest())
                    images.append(image); truths.append(truth); pending.append(row)
                    if len(pending) == 8:
                        flush()
                    del rendered, data
                if frame % 20 == 0 or frame == len(scene.views)-1:
                    flush(); stream.flush()
                    state = {'state': 'running', 'group': group, 'frame': frame+1, 'frames': len(scene.views),
                             'evaluated_rows': len(all_rows), 'seconds': time.monotonic()-started}
                    save(ROOT/'evaluation/STATUS.json', state); print(json.dumps(state), flush=True)
            flush(); del scene; torch.cuda.empty_cache()
        save(destination/'RASTERIZER_PARITY.json', parity)
        save(destination/'GEOMETRY_SAMPLES.json', geometry)
    after = {name: sha(Path(name)) for name in initial_hashes}
    assert initial_hashes == after, 'Evaluation must not modify model inputs'
    save(destination/'INPUT_INTEGRITY.json', {'before': initial_hashes, 'after': after, 'passed': True})
    assert len(all_rows) == 3*792*len(CASES)
    summary = []
    for group in ['baseline', 'joint', 'global_control']:
        for name, scope, ratio in CASES:
            for partition in ['all', 'bank', 'nonbank']:
                rows = [row for row in all_rows if row['group'] == group and row['case'] == name and
                        (partition == 'all' or row['bank'] == (partition == 'bank'))]
                assert rows
                result = dict(group=group, case=name, scope=scope, ratio=ratio, partition=partition, frames=len(rows))
                for key in ['psnr', 'ssim', 'lpips', 'photo_l1', 'K', 'candidate_N', 'projected_K', 'contributing_K']:
                    values = np.asarray([row[key] for row in rows], dtype=np.float64)
                    for label, value in [('mean', values.mean()), ('median', np.median(values)),
                                         ('p10', np.percentile(values, 10)), ('p90', np.percentile(values, 90))]:
                        result[key+'_'+label] = float(value)
                summary.append(result)
    save(destination/'SUMMARY.json', summary)
    with (destination/'SUMMARY.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    save(ROOT/'evaluation/STATUS.json', {'state': 'complete', 'groups': 3, 'frames_per_group': 792,
        'cases_per_group': len(CASES), 'rows': len(all_rows), 'seconds': time.monotonic()-started,
        'images_saved': 0, 'rasterizer_parity_checks': len(parity)})


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        save(ROOT/'evaluation/STATUS.json', {'state': 'failed', 'error': repr(error)})
        raise
