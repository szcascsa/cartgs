"""补算新联合模型0.08的全轨迹质量，合并其余已验证ratio；不训练。"""
import csv
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path

import evaluate_full_experiment as base

ROOT = base.ROOT
OUT = ROOT/'evaluation/ratio_quality_joint_20261105'
TRAIN = [.01, .05, .1, .3, .5, .7]
UNSEEN = [.03, .08, .2, .4, .6]
RATIOS = sorted(TRAIN+UNSEEN)


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
    for path in [previous/'PER_VIEW.csv', Path(__file__), Path(base.__file__), binding_path]+list((ROOT/'evaluation/python').rglob('*.py')):
        protected[str(path)] = base.sha(path)
    assert all(base.sha(Path(p)) == h for p, h in protected.items())
    base.save(OUT/'INPUTS_BEFORE.json', protected)
    spec = importlib.util.spec_from_file_location('fr1xyz_production_counts', binding_path)
    binding = importlib.util.module_from_spec(spec); spec.loader.exec_module(binding)
    scene = base.BudgetScene(checkpoint, '/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz', 5000, 0, None, 'pytorch')
    assert len(scene.views) == 792 and scene.renderer.total == 13298
    all_rows = [r for r in csv.DictReader((previous/'PER_VIEW.csv').open()) if r['group'] == 'joint' and
                (r['scope'] == 'full' or float(r['ratio']) in RATIOS)]
    assert len(all_rows) == 16632
    full = {int(r['frame']): r for r in all_rows if r['scope'] == 'full'}
    lpips = base.LearnedPerceptualImagePatchSimilarity(normalize=True, reduction='none').cuda()
    predictions, targets, pending, parity = [], [], [], []
    with (OUT/'PER_VIEW.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=base.FIELDS); writer.writeheader(); writer.writerows(all_rows)

        def flush():
            if not pending:
                return
            images, truths = base.torch.stack(predictions), base.torch.stack(targets)
            difference = images-truths
            quality = -10*base.torch.log10(difference.square().mean((1, 2, 3)))
            structure = base.structural_similarity_index_measure(images, truths, data_range=1., reduction='none')
            perceptual = lpips(images, truths); lpips.reset()
            photo = difference.abs().mean((1, 2, 3))
            metrics = base.torch.stack([quality, structure, perceptual.reshape(-1), photo], 1)
            assert base.torch.isfinite(metrics).all()
            for row, values in zip(pending, metrics.cpu().tolist()):
                row.update(zip(['psnr', 'ssim', 'lpips', 'photo_l1'], values))
                writer.writerow(row); all_rows.append(row)
            predictions.clear(); targets.clear(); pending.clear()

        for frame, view in enumerate(scene.views):
            assert full[frame]['image'] == view['image'] and float(full[frame]['timestamp']) == view['timestamp']
            c = scene.calibration
            camera = base.MiniCam(c['width'], c['height'], base.focal2fov(c['fx'], c['width']),
                base.focal2fov(c['fy'], c['height']), base.np.linalg.inv(base.np.asarray(view['pose'])))
            truth = scene.ground_truth(frame)
            assert truth is not None
            for scope in ['global', 'view']:
                data, ids, stats = scene.renderer.prepare(None, global_ratio=.08) if scope == 'global' else scene.renderer.prepare(camera, ratio_budget=.08)
                if scope == 'view':
                    assert len(ids) == min(math.floor(.08*stats['candidate_gaussians']), stats['eligible_candidates'])
                output = base.counted_render(binding, camera, data, scene.background)
                assert 0 <= int((output[5] > 0).sum()) <= int((output[3] > 0).sum()) <= len(ids)
                if frame in [0, 396, 791]:
                    normal = base.render_prepared(camera, data, scene.background, return_radii=True)
                    assert base.torch.equal(normal['render'].view(base.torch.int32), output[2].view(base.torch.int32))
                    assert base.torch.equal(normal['radii'], output[3])
                    parity.append({'frame': frame, 'scope': scope, 'ratio': .08, 'color_radii_bitwise': True})
                row = dict(group='joint', case=scope+'_0.08', scope=scope, ratio=.08, ratio_split='unseen',
                    frame=frame, timestamp=view['timestamp'], image=view['image'], bank=full[frame]['bank'],
                    N=scene.renderer.total, candidate_N=stats['candidate_gaussians'], eligible_N=stats['eligible_candidates'],
                    requested_K=stats['requested_budget'], K=len(ids), projected_K=int((output[3] > 0).sum()),
                    contributing_K=int((output[5] > 0).sum()), budget_shortfall=stats['budget_shortfall'],
                    selection_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest())
                predictions.append(output[2].clamp(0, 1)); targets.append(truth); pending.append(row)
                if len(pending) == 8:
                    flush()
                del output, data
            if frame % 40 == 0 or frame == 791:
                flush(); stream.flush()
                state = dict(state='running', frame=frame+1, frames=792, new_rows=len(all_rows)-16632, seconds=time.monotonic()-started)
                base.save(OUT/'STATUS.json', state); print(json.dumps(state), flush=True)
        flush()
    after = {p: base.sha(Path(p)) for p in protected}
    base.save(OUT/'INPUTS_AFTER.json', after); assert protected == after
    assert len(all_rows) == 18216
    base.save(OUT/'VALIDATION.json', parity)
    base.save(OUT/'STATUS.json', dict(state='complete', model='joint', frames=792, rows=len(all_rows),
        reused_rows=16632, new_renders=1584, parity_checks=len(parity), training_steps=0, images_saved=0,
        inputs_unchanged=True, train_ratios=TRAIN, unseen_ratios=UNSEEN,
        requested_unseen_overlap=[.7], seconds=time.monotonic()-started))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        OUT.mkdir(exist_ok=True)
        base.save(OUT/'STATUS.json', {'state': 'failed', 'error': repr(error)})
        raise
