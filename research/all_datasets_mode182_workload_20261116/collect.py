"""mode182真实逐像素工作量；只读冻结模型，计数不用于测速。"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent/'all_datasets_joint182_ratio_speedup_20261115'
sys.path.insert(0, str(PRIOR))
import worker_passes as old
h, p, torch, np = old.h, old.p, old.torch, old.np
release = old.release
METHODS, RATIOS, METHOD_RATIO = old.METHODS, old.RATIOS, old.METHOD_RATIO
RELEASE = old.RELEASE
COUNTERS = ['gaussian_tile_instances', 'potential_valid_pixel_pairs', 'scheduled_pixel_pair_slots',
    'active_pixel_tests', 'effective_alpha_blends', 'fallback_active_tests', 'overflow_tiles',
    'nonempty_tiles', 'covered_pixels', 'terminated_pixels', 'tile_count_square_sum', 'tile_count_sum']


def save(path, value):
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    pending.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def reference(scene):
    """加载已接受的三轮独占GPU计时及稳定选点身份。"""
    folder = PRIOR/'runs'/scene['slug']
    folders = [folder/'formal']
    validation = json.loads((folders[0]/'VALIDATION.json').read_text())
    if validation['repeats'] == 1:
        folders.append(folder/'remaining')
    records = {}
    for run in folders:
        assert json.loads((run/'VALIDATION.json').read_text())['passed']
        with (run/'TIMINGS.csv').open(newline='') as stream:
            for row in csv.DictReader(stream):
                key = int(row['frame']), row['method']
                identity = tuple(int(row[x]) for x in ['N', 'M', 'eligible', 'K'])+(row['ids_sha256'],)
                if key not in records:
                    records[key] = dict(identity=identity, times=[], repeats=[])
                record = records[key]
                assert record['identity'] == identity
                record['times'].append(float(row['raster_ms']))
                record['repeats'].append(int(row['repeat']))
    assert len(records) == scene['coverage']['evaluated_views']*len(METHODS)
    for record in records.values():
        assert sorted(record['repeats']) == [0, 1, 2]
        record['median_ms'] = float(np.median(record.pop('times')))
    return records


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    scene = next(s for s in json.loads((PRIOR/'SCENES_FROZEN.json').read_text()) if s['slug'] == args.scene)
    output = ROOT/'runs'/scene['slug']/('smoke' if args.smoke else 'formal')
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    h.configure()
    build = json.loads((ROOT/'BUILD_VALIDATION.json').read_text())
    native = release.extension()
    diagnostic = h.verified('mycartgs_mode182_workload_v1', Path(build['binary']), build['sha256'])
    def load_views(checkpoint, dataset):
        assert Path(checkpoint).resolve() == Path(scene['checkpoint']).resolve()
        calibration, views, coverage = old.views_for(scene)
        assert coverage == scene['coverage']
        return calibration, views
    old.budget_viewer.load_views = load_views
    protected = dict(scene['input_sha256'])
    protected.update(json.loads((ROOT/'BUILD_INPUTS.json').read_text()))
    for path in [Path(build['binary']), Path(__file__), ROOT/'build_diagnostic.py']:
        protected[str(path)] = sha(path)
    assert all(sha(path) == value for path, value in protected.items())
    source = p.TeacherSamples(Path(scene['checkpoint']), Path(scene['dataset']))
    student, _ = old.load_student(Path(scene['student']), 'cuda', source.fingerprint)
    student.eval().requires_grad_(False)
    engine = release.JointReleaseRenderer(source, student, **old.SETTINGS)
    original_raster = engine.native
    engine.native = diagnostic.WorkloadRasterizer(512, True, True, 15, 200, 64)
    engine.handle = engine.native._shared_handle()
    camera_spec = importlib.util.spec_from_file_location('workload_camera', RELEASE/'source_joint_release/preparation_camera.py')
    camera_api = importlib.util.module_from_spec(camera_spec)
    camera_spec.loader.exec_module(camera_api)
    camera = camera_api.NativeCameraPreparer(p.MiniCam, native)
    background = source.scene.background
    count = len(source.scene.views)
    assert count == scene['coverage']['evaluated_views'] and engine.network.total == scene['gaussians']
    W, H = scene['calibration']['width'], scene['calibration']['height']
    tiles = math.ceil(W/16)*math.ceil(H/16)
    refs = reference(scene)
    validation_frames = set(map(int, np.linspace(0, count-1, 4)))
    frames = sorted(validation_frames) if args.smoke else range(count)
    model_identity = dict(scene=scene, source_fingerprint=source.fingerprint,
        renderer=old.SETTINGS, diagnostic=build, protected_before=protected,
        timing_source=str(PRIOR), times='previous three accepted rounds, per-view median; counters never timed',
        counters=COUNTERS, saved_render_images=0, training=False, production_changed=False)
    save(output/'PROTOCOL.json', model_identity)
    checks, rows, blends_total = [], 0, 0
    stream = (output/'WORKLOAD.csv').open('x', newline='', encoding='utf-8')
    writer = None

    def render(raster, cam, packet):
        x, empty = packet['inputs'], background.new_empty(0)
        return raster.render(background, x['means3D'], empty, x['opacities'], x['scales'], x['rotations'],
            1., empty, cam.world_view_transform, cam.full_proj_transform, math.tan(cam.FoVx*.5),
            math.tan(cam.FoVy*.5), H, W, x['dc'], x['shs'], packet['sh_degree'], cam.camera_center, 5, 256, 32, False)

    for index, frame in enumerate(frames):
        if index % 100 == 0:
            assert shutil.disk_usage(ROOT).free >= 5*1024**3, '可用磁盘不足5GiB'
        for method in METHODS:
            ratio = 1. if method == 'ours_full' else METHOD_RATIO[method]
            condition = 'full' if method == 'ours_full' else f'local{ratio:g}'
            image, radii, ids, stats = engine.render_from_source(source, frame, background, condition, camera)
            diagnostic_image, total, maximum = engine.native.workload()
            evidence = torch.stack([maximum,
                (image.view(torch.int32) != diagnostic_image.view(torch.int32)).sum(),
                (~torch.isfinite(diagnostic_image)).sum(), (radii > 0).sum()])
            values = torch.cat([total, evidence]).cpu().tolist()
            totals = dict(zip(COUNTERS, map(int, values[:12])))
            max_count, differing, nonfinite, projected = map(int, values[12:])
            assert differing == nonfinite == 0, (frame, method, differing, nonfinite)
            assert 0 <= totals['effective_alpha_blends'] <= totals['active_pixel_tests'] <= totals['potential_valid_pixel_pairs']
            assert totals['active_pixel_tests'] <= totals['scheduled_pixel_pair_slots']
            assert totals['covered_pixels'] <= H*W and totals['terminated_pixels'] <= totals['covered_pixels']
            assert totals['gaussian_tile_instances'] == totals['tile_count_sum']
            digest = hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
            ref = refs[frame, method]
            n, m, eligible, k, expected_digest = ref['identity']
            assert len(ids) == k and digest == expected_digest, (frame, method, '先前测速选点身份不一致')
            assert m == (engine.network.total if method == 'ours_full' else stats['candidate_gaussians'])
            if method != 'ours_full':
                assert k == min(int(np.floor(ratio*m)), eligible)
            if frame in validation_frames:
                cam = camera.from_source(source, frame)
                if method == 'ours_full':
                    packet, split_ids = engine.network.full()
                else:
                    packet, split_ids, _ = engine.network.prepare(cam, ratio_budget=ratio)
                assert torch.equal(split_ids, ids)
                expected_image, expected_radii = render(original_raster, cam, packet)
                assert p.bits_equal(expected_image, image), (frame, method, '生产RGB差异')
                assert p.bits_equal(expected_radii, radii), (frame, method, '生产radii差异')
                old_image, flags, interactions = engine.native.diagnose()
                assert p.bits_equal(old_image, image), (frame, method, '已有诊断RGB差异')
                assert interactions.item() == totals['effective_alpha_blends'], (frame, method, '已有诊断计数差异')
                snapshot_counts = engine.native.snapshot()[3].long()
                assert snapshot_counts.sum().item() == totals['gaussian_tile_instances']
                assert snapshot_counts.max().item() == max_count
                checks.append(dict(frame=frame, method=method, RGB_bitwise=True, radii_bitwise=True,
                    selected_ids_bitwise=True, old_counter_exact=True, snapshot_counts_exact=True,
                    alpha_contributing_gaussians=int(flags.count_nonzero())))
            tile_mean = totals['gaussian_tile_instances']/tiles
            row = dict(scene=scene['id'], dataset=scene['dataset_kind'], scene_name=scene['scene'], mode=scene['mode'],
                frame=frame, trajectory_index=source.scene.views[frame]['trajectory_index'],
                timestamp=source.scene.views[frame]['timestamp'], method=method, ratio=ratio,
                width=W, height=H, pixels=H*W, N=n, M=m, eligible=eligible, K=k,
                projected_gaussians=projected, tiles=tiles, **totals, max_tile_gaussians=max_count,
                mean_tile_gaussians=tile_mean,
                std_tile_gaussians=math.sqrt(max(0, totals['tile_count_square_sum']/tiles-tile_mean**2)),
                alpha_blends_per_pixel=totals['effective_alpha_blends']/(H*W),
                active_tests_per_pixel=totals['active_pixel_tests']/(H*W),
                scheduled_slots_per_pixel=totals['scheduled_pixel_pair_slots']/(H*W),
                prior_raster_median_ms=ref['median_ms'], timing_repeats=3, ids_sha256=digest,
                diagnostic_RGB_bitwise=True)
            if writer is None:
                writer = csv.DictWriter(stream, list(row))
                writer.writeheader()
            writer.writerow(row)
            rows += 1
            blends_total += totals['effective_alpha_blends']
        if (index+1) % 100 == 0 or index+1 == len(frames):
            stream.flush()
            status = dict(state='collecting', frames=index+1, expected_frames=len(frames), rows=rows,
                elapsed_s=time.time()-started, last_frame=frame, effective_alpha_blends=blends_total)
            save(output/'STATUS.json', status)
            print(json.dumps(status), flush=True)
    stream.close()
    assert rows == len(frames)*len(METHODS)
    assert all(sha(path) == value for path, value in protected.items()), '保护文件发生变化'
    save(output/'REFERENCE_CHECKS.json', checks)
    validation = dict(passed=True, rows=rows, frames=len(frames), methods=METHODS,
        all_rows_diagnostic_RGB_bitwise=True, all_rows_prior_ids_K_exact=True,
        production_RGB_radii_ids_checked=len(checks), existing_counter_checked=len(checks),
        protected_files=len(protected), protected_unchanged=True, saved_render_images=0,
        training=False, production_changed=False, elapsed_s=time.time()-started,
        workload_csv_sha256=sha(output/'WORKLOAD.csv'),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved())
    save(output/'VALIDATION.json', validation)
    save(output/'STATUS.json', dict(state='complete', **validation))
    print('COMPLETE', json.dumps(validation), flush=True)


if __name__ == '__main__':
    main()
