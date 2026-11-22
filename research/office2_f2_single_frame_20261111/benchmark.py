"""固定Tensor＋新准备D，完整轨迹配对比较旧G与F2；CaRtGS原版入口不替换。"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'office2_single_frame_curves_20261111'
F2_ROOT = HERE.parent/'render_count_scaling_20261111_2022'
spec = importlib.util.spec_from_file_location('office2_original_protocol', OLD/'benchmark.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
h, p, torch, np = b.h, b.p, b.torch, b.np
import cv2


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--attempt', default='formal')
    parser.add_argument('--timing-only', action='store_true')
    args = parser.parse_args()
    output = HERE/('smoke' if args.smoke else args.attempt)
    output.mkdir(exist_ok=False)
    save = lambda name, obj: p.save(output/name, obj)
    write = lambda name, rows: p.write_csv(output/name, rows)
    c, d, cm, dm = h.configure()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    frozen_path = F2_ROOT/'background_stage/reports/FROZEN.json'
    frozen = json.loads(frozen_path.read_text())
    assert h.sha(frozen['binary']) == frozen['binary_sha256']
    for name, sha in frozen['sources'].items():
        assert h.sha(F2_ROOT/'background_stage'/name) == sha, name
    sys.path.insert(0, str(F2_ROOT))
    spec = importlib.util.spec_from_file_location('office2_f2_runtime', F2_ROOT/'background_runtime.py')
    f2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f2)
    f2.extension = lambda: h.verified('mycartgs_render_background_scaling_v1', Path(frozen['binary']), frozen['binary_sha256'])
    before = {name: h.sha(name) for name in json.loads((OLD/'reports/PROTECTED_AFTER.json').read_text())}
    for path in [frozen_path, Path(frozen['binary']), F2_ROOT/'background_runtime.py', F2_ROOT/'scaling_runtime.py']:
        before[str(path)] = h.sha(path)
    for name, sha in frozen['sources'].items():
        before[str((F2_ROOT/'background_stage'/name).resolve())] = sha
    save('PROTECTED_BEFORE.json', before)
    scene, source = p.load_scene(b.prior.SLUG)
    student, student_path, payload = p.student_for(source, b.prior.SLUG)
    engine = h.engine(source, student, 'D')
    camera = h.Camera(source, 'D', dm, b.SIZE)
    cp = b.prior.VANILLA/'model/15900_shutdown/ply'
    ply, = list((cp/'point_cloud').glob('iteration_*/point_cloud.ply'))
    original_pc = b.prior.GaussianModel(3)
    original_pc.load_ply(str(ply), 'cuda')
    for name in ['_xyz', '_scaling', '_rotation', '_opacity', '_features_dc', '_features_rest']:
        getattr(original_pc, name).requires_grad_(False)
    assert engine.total == 43197 and len(original_pc.get_xyz) == 84203
    records = np.atleast_2d(np.loadtxt(b.prior.VANILLA/'model/CameraTrajectory_TUM.txt'))
    stamps = {float(row[0]): row for row in records}
    poses = {}
    for frame, view in enumerate(source.scene.views):
        row = stamps[float(view['timestamp'])]
        pose = np.eye(4)
        pose[:3, :3] = b.Rotation.from_quat(row[4:8]).as_matrix()
        pose[:3, 3] = row[1:4]
        poses[frame] = pose
    n = len(source.scene.views)
    frames = [0, 500, 722, 741, 1000, 1999] if args.smoke else list(range(n))
    renderers = {'G': p.create_renderer(camera.create(0), source.scene.background, engine.total, use_graph=True),
                 'F2': f2.Renderer()}
    original = list(csv.DictReader((OLD/'reports/QUALITY_COUNTS.csv').open()))
    expected = {(row['method'], int(row['frame'])): row for row in original}
    save('PROTOCOL.json', dict(dataset='Replica/office2/rgbd', width=2064, height=2208, renders_per_frame=1,
        frames=len(frames), trajectory_frames=n, rounds=1 if args.smoke else 3,
        methods=b.METHODS, versions=['original', 'G', 'F2'], student_parameters=3534,
        preparation='same frozen Tensor Core D and new preparation, independent current camera and networks per request',
        comparison='same live packet G/F2, interleaved reversed version order; original CaRtGS uses original rasterizer',
        quality='all-frame current RGB/radii bitwise G=F2=diagnostic; current selected IDs, counts and PSNR match historical',
        timing='single render CUDA Event; independent prepare+render wall; cold plans and counters excluded',
        precision='unchanged Tensor network; F2 rasterizer FP32; no approximate rendering',
        excluded='GT, counters, model loads, cold graph compilation, output files, display, SLAM',
        hardware=torch.cuda.get_device_name(), training=False, production_changed=False, saved_images=0))
    save('MODEL_IDENTITIES.json', dict(scene=scene, vanilla_checkpoint=str(cp), vanilla_N=84203, ours_N=43197,
        student=str(student_path), student_sha256=h.sha(student_path), D_binary=d['binary'], D_sha256=d['binary_sha256'],
        original_rasterizer_sha256=b.prior.ORIGINAL_SHA, F2_binary=frozen['binary'], F2_sha256=frozen['binary_sha256'],
        F2_freeze_sha256=h.sha(frozen_path), F2_mode=18))
    spec = importlib.util.spec_from_file_location('fr1xyz_production_counts', b.COUNTER)
    binding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(binding)

    def prepare(frame, method):
        if method == 'cartgs_full':
            cam = p.MiniCam(*b.SIZE, camera.fx, camera.fy, np.linalg.inv(poses[frame]))
            packet = b.renderer.prepare_render_data(original_pc)
            return cam, packet, None, dict(K=84203, M=84203, requested_K=84203)
        cam = camera.create(frame)
        kwargs = dict(ratio_budget=.5) if method == 'ratio_50' else dict(budget=int(method[6:]))
        packet, ids, stats = engine.prepare(cam, **kwargs)
        return cam, packet, ids, dict(K=len(ids), M=stats['candidate_gaussians'], requested_K=stats['requested_budget'])

    def raster(cam, packet, version):
        if version == 'original':
            out = p.render_prepared(cam, packet, source.scene.background, return_radii=True)
            return out['render'], out['radii']
        return renderers[version].render(cam, packet, source.scene.background)

    quality, checks = [], []
    if args.timing_only:
        valid = json.loads((HERE/'formal/VALIDATION.json').read_text())
        assert valid['passed'] and valid['G_F2_RGB_radii_bitwise'] and valid['frames'] == n
        quality = list(csv.DictReader((HERE/'formal/QUALITY_COUNTS.csv').open()))
        checks = list(csv.DictReader((HERE/'formal/CHECKS.csv').open()))
        save('REUSED_QUALITY.json', dict(source=str(HERE/'formal'),
            validation_sha256=h.sha(HERE/'formal/VALIDATION.json'), timing_reused=False))
    for index, frame in enumerate([] if args.timing_only else frames):
        view = source.scene.views[frame]
        image = cv2.imread(view['image'], cv2.IMREAD_COLOR)
        assert image is not None
        truth = cv2.cvtColor(cv2.resize(image, b.SIZE, interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        gt = torch.from_numpy(truth).to('cuda').permute(2, 0, 1)/255.
        for method in b.METHODS:
            cam, packet, ids, stats = prepare(frame, method)
            ref = expected[method, frame]
            assert all(stats[key] == int(ref[key]) for key in ['K', 'M', 'requested_K'])
            digest = 'all_map' if ids is None else hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
            assert digest == ref['ids_sha256']
            counted = b.counted(binding, cam, packet, source.scene.background)
            psnr = float(-10*torch.log10((counted[2].clamp(0, 1)-gt).square().mean()))
            assert psnr == float(ref['psnr']), (frame, method, psnr, ref['psnr'])
            projected, contributed = int((counted[3] > 0).sum()), int((counted[5] > 0).sum())
            assert projected == int(ref['projected_K']) and contributed == int(ref['contributing_K'])
            for version in ['original'] if method == 'cartgs_full' else ['G', 'F2']:
                rgb, radii = raster(cam, packet, version)
                assert p.bits_equal(rgb, counted[2]) and p.bits_equal(radii, counted[3]), (frame, method, version)
                checks.append(dict(frame=frame, method=method, version=version, rgb_radii_bitwise=True))
            quality.append(dict(ref))
        if index % 100 == 0 or index == len(frames)-1:
            write('QUALITY_COUNTS.csv', quality)
            write('CHECKS.csv', checks)
            print('QUALITY', index+1, len(frames), flush=True)
    write('QUALITY_COUNTS.csv', quality)
    write('CHECKS.csv', checks)
    # 独立测速进程重建全部形状计划；不复用之前的网络输出或Mask。
    for frame in frames if args.timing_only else np.linspace(0, n-1, 24, dtype=int):
        for method in b.METHODS:
            cam, packet, ids, stats = prepare(int(frame), method)
            if args.timing_only and ids is not None:
                assert hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest() == expected[method, int(frame)]['ids_sha256']
            for version in ['original'] if method == 'cartgs_full' else ['G', 'F2']:
                raster(cam, packet, version)
    rows = []
    for repeat in range(1 if args.smoke else 3):
        ordered = list(frames)
        if repeat == 1:
            ordered.reverse()
        elif repeat == 2:
            np.random.default_rng(0).shuffle(ordered)
        for index, frame in enumerate(ordered):
            for method in p.rotate(b.METHODS, repeat+index):
                torch.cuda.synchronize()
                tick = time.perf_counter_ns()
                cam, packet, ids, stats = prepare(frame, method)
                torch.cuda.synchronize()
                prepare_ms = (time.perf_counter_ns()-tick)/1e6
                versions = ['original'] if method == 'cartgs_full' else (['G', 'F2'] if (repeat+index)%2 == 0 else ['F2', 'G'])
                for version in versions:
                    torch.cuda.synchronize()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    result = raster(cam, packet, version)
                    end.record()
                    end.synchronize()
                    raster_ms = start.elapsed_time(end)
                    torch.cuda.synchronize()
                    tick = time.perf_counter_ns()
                    current_cam, current_packet, current_ids, current_stats = prepare(frame, method)
                    result = raster(current_cam, current_packet, version)
                    torch.cuda.synchronize()
                    total_ms = (time.perf_counter_ns()-tick)/1e6
                    assert current_stats == stats
                    rows.append(dict(frame=frame, method=method, version=version, repeat=repeat,
                        order=index, **stats, prepare_ms=prepare_ms, raster_ms=raster_ms, total_ms=total_ms))
            if index % 200 == 0 or index == len(ordered)-1:
                write('TIMINGS.csv', rows)
                print('TIMING', repeat, index+1, len(ordered), flush=True)
    after = {name: h.sha(name) for name in before}
    changed = [name for name in before if before[name] != after[name]]
    save('PROTECTED_AFTER.json', after)
    result = dict(passed=not changed, changed=changed, protected_files=len(before), frames=len(frames),
        quality_rows=len(quality), timing_rows=len(rows), parity_checks=len(checks),
        previous_ids_K_M_counts_psnr_exact=True, G_F2_RGB_radii_bitwise=True,
        renders_per_frame=1, render_images_saved=0, training=False, production_changed=False,
        F2_stats=renderers['F2'].native.stats())
    save('VALIDATION.json', result)
    assert result['passed']
    print('COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
