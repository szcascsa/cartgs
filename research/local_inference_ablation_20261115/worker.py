"""固定模型与局部K，关闭全部补偿或随机选择后保留真实视角补偿。"""
from pathlib import Path
from contextlib import ExitStack
import argparse
import ast
import csv
import fcntl
import hashlib
import math
import os
import sys
import time

from common import ROOT, RATIOS, TRAIN, VARIANTS, SOURCE, records, sha, load, write


def setup(items, folder):
    from evo.tools.settings import SETTINGS
    SETTINGS.plot_backend = 'Agg'
    command = load(Path(items[0]['source'])/'launch.json')['command']
    entry = next(i for i, arg in enumerate(command) if arg.endswith('/eval_joint_entry.py'))
    original = Path(command[entry]).parent
    snapshot = ROOT/'snapshots'/original.name/'python'
    script = snapshot/'run_metrics_only.py'
    sys.path[:0] = [str(snapshot), str(original/'evaluation_runtime')]
    import anchor_frustum
    from anchor_budget import FixedBudgetRenderer

    class LocalRenderer:
        def __init__(self, model, gaussians, ratio, config, temperature, protection):
            assert int(model.joint_view_conditioning_schema) == 2
            self.renderer = FixedBudgetRenderer(model, gaussians,
                dict(config, temperature=temperature, protection_enabled=protection), seed=0, backend='pytorch')

    anchor_frustum.AnchorFrustumRenderer = LocalRenderer
    args = command[entry+2:]
    args.remove('--joint-local-ratio')
    assert not any(flag in args for flag in ['--stereo-camera-config', '--stereo-native-maps'])
    args[args.index('--output_path')+1] = str(folder/'setup')
    args.append('--selector-anchor-frustum')
    sys.argv = [str(script), *args]
    marker = '    psnr_list, ssim_list, lpips_list, time_list = [], [], [], []'
    source = script.read_text()
    assert source.count(marker) == 1
    tree = ast.parse(source.split(marker)[0], filename=str(script))
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
        ['calc_psnr', 'calc_ssim', 'calc_lpips'] for t in node.targets)) and not (
        isinstance(node, ast.ImportFrom) and (node.module or '').startswith('torchmetrics.'))]
    namespace = dict(__name__='__main__', __file__=str(script))
    exec(compile(ast.fix_missing_locations(tree), str(script), 'exec'), namespace)
    current = load(folder/'setup/evaluation_frames.json')
    assert load(folder/'setup/image_preprocessing.json') == {'protocol': 'original_rgb'}
    protected = {SOURCE, script, Path(args[args.index('--selector-config')+1])}
    checkpoint = Path(args[args.index('--selector')+1]).parent
    protected.update(path for path in checkpoint.rglob('*') if path.is_file())
    protected.add(Path(args[0])/'CameraTrajectory_TUM.txt')
    for item in items:
        output = Path(item['source'])
        assert load(output/'evaluation_frames.json') == current
        assert len(current['gt_indices']) == int(item['frames'])
        protected.update(output/name for name in ['selection_hashes.json', 'metric_summary.json',
            'per_frame_metrics.csv', 'anchor_frustum_per_view.csv', 'launch.json', 'evaluation_frames.json'])
    protected.update(Path(module.__file__) for module in list(sys.modules.values())
        if getattr(module, '__file__', None) and Path(module.__file__).is_file()
        and (str(module.__file__).startswith(str(original)) or str(module.__file__).startswith(str(snapshot))))
    return namespace, {str(path): sha(path) for path in protected}


def equal_bits(a, b):
    import torch
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))


def packet_equal(a, b):
    import torch
    assert a['sh_degree'] == b['sh_degree']
    assert a['inputs'].keys() == b['inputs'].keys()
    for key, value in a['inputs'].items():
        other = b['inputs'][key]
        assert (value is None and other is None) or (
            isinstance(value, torch.Tensor) and equal_bits(value, other)), key


def packet(fixed, compact, rows, mask, ratio, camera, code, compensation):
    from renderer import prepare_render_data
    from selector import compensated_attributes
    attributes = None
    if compensation and bool(mask.any()):
        attributes = compensated_attributes(fixed.model, compact, mask, ratio, source_rows=rows,
                                            camera=camera, anchor_code=code)
        if attributes is not None:
            attributes['scaling'] = attributes['scaling'].log().exp()
            attributes['rotation'] = fixed.gaussians.rotation_activation(attributes['rotation'])
    return prepare_render_data(compact, mask, attributes)


def run(group, smoke=False, attempt=1):
    import cv2
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(.32)
    items = sorted([row for row in records() if row['group'] == group], key=lambda row: float(row['ratio']))
    assert [float(row['ratio']) for row in items] == RATIOS
    folder = ROOT/('smoke' if smoke else 'runs')/group/f'attempt_{attempt:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder/'COMPLETE.json').exists():
        print('ALREADY_COMPLETE', group, flush=True)
        return
    started = time.monotonic()
    ns, protected = setup(items, folder)
    write(folder/'INPUTS_BEFORE.json', protected)
    from utils import MiniCam
    from anchor_frustum import anchor_frustum_mask, compact_gaussians
    from anchor_budget import exact_budget_mask
    from selector import slot_eligibility
    from selector_sampling import gumbel_selection
    from renderer import render_prepared
    from torchmetrics.functional.image import structural_similarity_index_measure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from torchmetrics.image.psnr import PeakSignalNoiseRatio
    from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True, reduction='none').cuda()
    from cached_metric import CachedTargetLPIPS
    cached_lpips = CachedTargetLPIPS(lpips)
    single = dict(psnr=PeakSignalNoiseRatio(data_range=1.).cuda(),
                  ssim=StructuralSimilarityIndexMeasure(data_range=1.).cuda(),
                  lpips=LearnedPerceptualImagePatchSimilarity(normalize=True).cuda())
    fixed = ns['frustum_renderer'].renderer
    assert fixed.total == int(items[0]['N'])
    state = {name: value.detach().clone() for name, value in fixed.model.state_dict().items()}
    hashes = [load(Path(item['source'])/'selection_hashes.json') for item in items]
    history = [list(csv.DictReader((Path(item['source'])/'anchor_frustum_per_view.csv').open())) for item in items]
    historical_metrics = [list(csv.DictReader((Path(item['source'])/'per_frame_metrics.csv').open())) for item in items]
    frames = len(ns['associations'])
    assert all(len(h) == len(v) == len(m) == frames for h,v,m in zip(hashes, history, historical_metrics))
    anchors = fixed.model.anchors
    check_frames = {0, frames//2, frames-1}
    frame_indices = sorted(check_frames) if smoke else range(frames)
    batch_size = max(1, min(8, 2_000_000//(ns['width']*ns['height'])))
    fields = ['group', 'variant', 'ratio', 'ratio_kind', 'frame', 'trajectory_index', 'rgb_index',
              'rgb_timestamp', 'N', 'candidate_N', 'eligible_N', 'requested_K', 'K', 'budget_shortfall',
              'selection_sha256', 'selection_jaccard_vs_learned', 'psnr', 'ssim', 'lpips']
    sums = {(variant, ratio): np.zeros(6, dtype=np.float64) for variant in VARIANTS for ratio in RATIOS}
    extremes = {(variant, ratio): [fixed.total, 0] for variant in VARIANTS for ratio in RATIOS}
    pending, predictions, truths = [], [], []
    parity, metric_parity, packet_checks = [], [], []
    rows_written = 0
    peak_errors = dict(psnr=0., ssim=0., lpips=0.)
    metric_errors = dict(psnr=0., ssim=0., lpips=0.)
    np.save(folder/'RANDOM_PRIORITIES.npy', fixed.random_scores.cpu().numpy())
    with ExitStack() as streams, torch.no_grad():
        writers = {}
        for variant in VARIANTS:
            for ratio in RATIOS:
                path = folder/variant/f'ratio_{ratio:g}'/'per_frame_metrics.csv'
                path.parent.mkdir(parents=True, exist_ok=True)
                stream = streams.enter_context(path.open('w', newline=''))
                writer = csv.DictWriter(stream, fields)
                writer.writeheader()
                writers[variant, ratio] = (writer, stream)

        def flush():
            nonlocal rows_written
            if not pending:
                return
            prediction, target = torch.stack(predictions), torch.stack(truths)
            values = torch.stack([-10*torch.log10((prediction-target).square().mean((1,2,3))),
                structural_similarity_index_measure(prediction, target, data_range=1., reduction='none'),
                cached_lpips(prediction, truths, [row['frame'] for row in pending])], 1)
            lpips.reset()
            assert torch.isfinite(values).all(), 'NONFINITE_METRICS'
            for index, (row, metric) in enumerate(zip(pending, values.cpu().tolist())):
                row.update(zip(['psnr', 'ssim', 'lpips'], metric))
                if row['variant'] == 'reference_reproduction':
                    old = historical_metrics[RATIOS.index(row['ratio'])][row['frame']]
                    errors = {name: abs(row[name]-float(old[name])) for name in peak_errors}
                    for name in errors:
                        peak_errors[name] = max(peak_errors[name], errors[name])
                    assert errors['psnr'] < 1e-4 and errors['ssim'] < 2e-5 and errors['lpips'] < 2e-5, (group, row['ratio'], row['frame'], errors)
                    scalar_errors = {}
                    for name, module in single.items():
                        module.reset()
                        scalar_errors[name] = abs(float(module(prediction[index:index+1], target[index:index+1]))-row[name])
                        metric_errors[name] = max(metric_errors[name], scalar_errors[name])
                    assert scalar_errors['psnr'] < 1e-4 and scalar_errors['ssim'] < 2e-5 and scalar_errors['lpips'] < 2e-5
                    parity.append(dict(frame=row['frame'], ratio=row['ratio'], historical_error=errors))
                    metric_parity.append(dict(frame=row['frame'], ratio=row['ratio'], scalar_error=scalar_errors))
                else:
                    key = row['variant'], row['ratio']
                    writers[key][0].writerow(row)
                    sums[key] += [*metric, row['K'], row['candidate_N'], 1.]
                    extremes[key][0] = min(extremes[key][0], row['K'])
                    extremes[key][1] = max(extremes[key][1], row['K'])
                    rows_written += 1
            pending.clear(); predictions.clear(); truths.clear()

        for frame in frame_indices:
            trajectory, rgb = ns['associations'][frame]
            camera = MiniCam(ns['width'], ns['height'], ns['fovx'], ns['fovy'], np.linalg.inv(ns['poses'][trajectory]))
            image = cv2.imread(ns['gt_color_paths'][rgb], cv2.IMREAD_COLOR)
            assert image is not None
            truth = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).to('cuda').permute(2,0,1)/255.
            visible = anchor_frustum_mask(anchors.centers, anchors.offset_scales, camera.full_proj_transform, fixed.occupied)
            rows = torch.nonzero(visible[anchors.rows]).flatten()
            compact = compact_gaussians(fixed.gaussians, rows)
            eligible = slot_eligibility(fixed.model, compact.get_opacity, rows)
            population, available = len(rows), int(eligible.sum())
            code = fixed.model.joint_view_code(camera) if population and available else None
            previous_random = None
            for index, ratio in enumerate(RATIOS):
                budget = math.floor(ratio*population)
                target = min(budget, available)
                mask = torch.zeros(population, dtype=torch.bool, device=rows.device)
                if target >= available and available:
                    mask = eligible.clone()
                elif target:
                    _, _, logits = fixed.model.forward_view_selection_features(compact.get_xyz,
                        compact.get_scaling, compact.get_features_dc, compact.get_opacity,
                        fixed.gaussians.get_xyz.new_tensor([[ratio]]), camera, source_rows=rows,
                        return_logits=True, anchor_code=code)
                    assert torch.isfinite(logits).all()
                    mask = exact_budget_mask(gumbel_selection(logits, hard=False, noise=fixed.noise[rows]), eligible, budget)
                ids = rows[mask]
                digest = hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
                old = history[index][frame]
                assert digest == hashes[index][frame], (group, frame, ratio, '历史Selector ID不符')
                assert len(ids) == int(float(old['active_gaussians'])) == target
                assert population == int(float(old['candidate_gaussians']))
                assert available == int(float(old['eligible_candidates']))
                assert budget == int(float(old['requested_budget']))
                random_mask = exact_budget_mask(fixed.random_scores[rows], eligible, budget)
                random_ids = rows[random_mask]
                assert len(random_ids) == len(ids) == target
                assert not bool((random_mask & ~eligible).any())
                if previous_random is not None:
                    assert not bool((previous_random & ~random_mask).any()), '随机集合不嵌套'
                previous_random = random_mask
                intersection = int((mask & random_mask).sum())
                jaccard = intersection/max(1, 2*target-intersection)
                cases = [('selector_no_comp', mask, False, digest, 1.),
                         ('random_with_comp', random_mask, True,
                          hashlib.sha256(random_ids.cpu().numpy().tobytes()).hexdigest(), jaccard)]
                if frame in check_frames:
                    cases.append(('reference_reproduction', mask, True, digest, 1.))
                for variant, selected_mask, compensation, selection_hash, overlap in cases:
                    data = packet(fixed, compact, rows, selected_mask, ratio, camera, code, compensation)
                    assert data['inputs']['means3D'].shape[0] == target
                    assert all(torch.isfinite(value).all() for value in data['inputs'].values() if isinstance(value, torch.Tensor))
                    if frame in check_frames and variant != 'reference_reproduction':
                        expected, expected_ids, _ = fixed.prepare(camera, ratio_budget=ratio, view_conditioned=True,
                            selection='random' if variant == 'random_with_comp' else 'learned',
                            compensation=compensation, measure_time=False)
                        assert equal_bits(expected_ids, rows[selected_mask])
                        packet_equal(data, expected)
                        packet_checks.append(dict(frame=frame, ratio=ratio, variant=variant, packet_bitwise_equal=True))
                    row = dict(group=group, variant=variant, ratio=ratio, ratio_kind='train' if ratio in TRAIN else 'unseen',
                        frame=frame, trajectory_index=int(trajectory), rgb_index=int(rgb),
                        rgb_timestamp=float(ns['gt_tstamp'][rgb]), N=fixed.total, candidate_N=population,
                        eligible_N=available, requested_K=budget, K=target, budget_shortfall=budget-target,
                        selection_sha256=selection_hash, selection_jaccard_vs_learned=overlap)
                    pending.append(row)
                    predictions.append(render_prepared(camera, data, ns['background'])['render'].clamp(0,1))
                    truths.append(truth)
                    if len(pending) >= batch_size:
                        flush()
            if frame % 16 == 0 or frame == frames-1 or smoke:
                flush()
                for _, stream in writers.values():
                    stream.flush()
                progress = dict(state='running', group=group, frame=frame+1, frames=frames,
                    rows=rows_written, elapsed_seconds=time.monotonic()-started, updated_unix=time.time())
                write(folder/'STATUS.json', progress)
                print('PROGRESS', __import__('json').dumps(progress), flush=True)
        flush()
    after = {name: sha(name) for name in protected}
    assert protected == after, '源checkpoint或历史评价被改写'
    write(folder/'INPUTS_AFTER.json', after)
    assert all(equal_bits(value, fixed.model.state_dict()[name]) for name, value in state.items())
    evaluated = len(frame_indices)
    assert rows_written == evaluated*22
    results = []
    for variant in VARIANTS:
        for index, ratio in enumerate(RATIOS):
            values = sums[variant, ratio]
            assert int(values[5]) == evaluated
            old = items[index]
            result = dict(group=group, variant=variant, ratio=ratio, ratio_kind='train' if ratio in TRAIN else 'unseen',
                frames=evaluated, N=fixed.total, K_mean=values[3]/evaluated,
                K_min=extremes[variant, ratio][0], K_max=extremes[variant, ratio][1],
                candidate_N_mean=values[4]/evaluated, seed=0, scope='view_ratio',
                selection='learned' if variant == 'selector_no_comp' else 'random',
                compensation=variant == 'random_with_comp', compensator_real_view=variant == 'random_with_comp',
                baseline_source=old['source'], image_protocol='original_rgb', training_steps=0, images_saved=0,
                output=str(folder/variant/f'ratio_{ratio:g}'))
            for column, name in enumerate(['psnr', 'ssim', 'lpips']):
                result[name] = values[column]/evaluated
                result['baseline_'+name] = float(old[name])
                result['delta_'+name] = result[name]-float(old[name])
            if not smoke:
                assert abs(result['K_mean']-float(old['K_mean'])) < 1e-9
            write(Path(result['output'])/'metric_summary.json', result)
            results.append(result)
    complete = dict(complete=not smoke, smoke=smoke, group=group, frames=evaluated, total_frames=frames,
        results=results, source_unchanged=True, model_state_bitwise_unchanged=True, random_nested=True,
        all_frame_ids_and_K_verified=True, packet_checks=packet_checks, reference_checks=parity,
        batch_metric_checks=metric_parity, historical_max_error=peak_errors, scalar_metric_max_error=metric_errors,
        training_steps=0, images_saved=0, seconds=time.monotonic()-started, batch_size=batch_size, lpips_target_cache=True, worker_sha256=sha(__file__),
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    write(folder/('SMOKE.json' if smoke else 'COMPLETE.json'), complete)
    write(folder/'STATUS.json', dict(state='complete', group=group, frames=evaluated, rows=rows_written, updated_unix=time.time()))
    print('GROUP_COMPLETE', group, 'frames', evaluated, 'seconds', time.monotonic()-started, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    run(args.group, args.smoke, args.attempt)
