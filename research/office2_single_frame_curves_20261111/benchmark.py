"""office2完整轨迹、单次高分辨率渲染：原版与Tensor＋新准备D。"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent/'quest3_office2_tensor_vs_cartgs_20261111'
spec = importlib.util.spec_from_file_location('single_frame_prior', PRIOR/'benchmark.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
h, p, torch, np = prior.h, prior.p, prior.torch, prior.np
import cv2
import renderer
import diff_gaussian_rasterization
from scipy.spatial.transform import Rotation

METHODS = ['cartgs_full', 'ratio_50', 'fixed_2000', 'fixed_4000']
SIZE = (2064, 2208)
COUNTER = Path('/home/prmi/zwj/experiments/fr1xyz_viewer_budget_comparison_20261105/build/fr1xyz_production_counts.so')


def counted(binding, camera, packet, background):
    inputs = packet['inputs']
    empty = torch.Tensor([])
    value = lambda name: inputs[name] if inputs[name] is not None else empty
    return binding.render_with_counts(background, value('means3D'), value('colors_precomp'),
        value('opacities'), value('scales'), value('rotations'), 1., value('cov3D_precomp'),
        camera.world_view_transform, camera.full_proj_transform,
        math.tan(camera.FoVx*.5), math.tan(camera.FoVy*.5), camera.image_height, camera.image_width,
        value('dc'), value('shs'), packet['sh_degree'], camera.camera_center,
        False, False, True, binding.Mode.Inference)


@torch.no_grad()
def main():
    args = argparse.ArgumentParser()
    args.add_argument('--smoke', action='store_true')
    args = args.parse_args()
    reports = HERE/('smoke' if args.smoke else 'reports')
    reports.mkdir(exist_ok=False)
    save = lambda name, value: p.save(reports/name, value)
    write = lambda name, rows: p.write_csv(reports/name, rows)
    c, d, cm, dm = h.configure()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    assert h.sha(diff_gaussian_rasterization._C.__file__) == prior.ORIGINAL_SHA
    before = json.loads((PRIOR/'reports/PROTECTED_AFTER.json').read_text())
    for name, expected in before.items():
        assert h.sha(name) == expected, name
    before[str(COUNTER)] = h.sha(COUNTER)
    save('PROTECTED_BEFORE.json', before)
    scene, source = p.load_scene(prior.SLUG)
    student, student_path, payload = p.student_for(source, prior.SLUG)
    engine = h.engine(source, student, 'D')
    camera = h.Camera(source, 'D', dm, SIZE)
    cp = prior.VANILLA/'model/15900_shutdown/ply'
    ply, = list((cp/'point_cloud').glob('iteration_*/point_cloud.ply'))
    original_pc = prior.GaussianModel(3)
    original_pc.load_ply(str(ply), 'cuda')
    for name in ['_xyz', '_scaling', '_rotation', '_opacity', '_features_dc', '_features_rest']:
        getattr(original_pc, name).requires_grad_(False)
    assert engine.total == 43197 and len(original_pc.get_xyz) == 84203
    records = np.atleast_2d(np.loadtxt(prior.VANILLA/'model/CameraTrajectory_TUM.txt'))
    by_timestamp = {float(row[0]): row for row in records}
    assert len(by_timestamp) == len(source.scene.views)
    poses = {}
    for frame, view in enumerate(source.scene.views):
        row = by_timestamp[float(view['timestamp'])]
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix()
        pose[:3, 3] = row[1:4]
        poses[frame] = pose
    all_frames = list(range(len(source.scene.views)))
    frames = [all_frames[0], all_frames[len(all_frames)//2], all_frames[-1]] if args.smoke else all_frames
    render = p.create_renderer(camera.create(frames[0]), source.scene.background, engine.total, use_graph=True)
    spec = importlib.util.spec_from_file_location('fr1xyz_production_counts', COUNTER)
    binding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(binding)
    save('PROTOCOL.json', dict(dataset='Replica/office2/rgbd', width=SIZE[0], height=SIZE[1],
        renders_per_frame=1, frames=len(frames), trajectory_frames=len(all_frames), methods=METHODS,
        rounds=1 if args.smoke else 3, hardware=torch.cuda.get_device_name(),
        ours='D: distilled 3534-parameter Compensator, Tensor Core, new preparation, G rasterizer',
        ratio='0.5 times current nominal Anchor frustum candidate count, strict K',
        fixed_budget='2000/4000 submitted Gaussians, clamped only if eligible candidate pool is insufficient',
        count='distinct Gaussians with at least one accepted Alpha-compositing pixel; diagnostic counts excluded from timing',
        quality='per-frame PSNR against original RGB resized to 2064x2208 with OpenCV INTER_LINEAR; clamp render to [0,1]',
        views='same RGB timestamps, each learned map uses own stored SLAM pose',
        raster_event='single render, camera and attributes already prepared; CUDA Event, no network',
        preparation='current camera and attribute/Selector/Compensator preparation, synchronized wall time',
        pipeline='independent single prepare+render measurement, no intermediate synchronization',
        excluded='loading, cold graph creation, GT, counters, file output, display, SLAM',
        training=False, production_changed=False, render_images_saved=0))
    save('MODEL_IDENTITIES.json', dict(scene=scene, ours_N=engine.total, vanilla_N=len(original_pc.get_xyz),
        vanilla_checkpoint=str(cp), student=str(student_path), student_sha256=h.sha(student_path),
        D_binary=d['binary'], D_binary_sha256=d['binary_sha256'],
        original_rasterizer=str(diff_gaussian_rasterization._C.__file__), original_sha256=prior.ORIGINAL_SHA,
        diagnostic_counter=str(COUNTER), diagnostic_counter_sha256=h.sha(COUNTER)))

    def prepare(frame, method):
        if method == 'cartgs_full':
            cam = p.MiniCam(*SIZE, camera.fx, camera.fy, np.linalg.inv(poses[frame]))
            packet = renderer.prepare_render_data(original_pc)
            return cam, packet, None, dict(K=len(original_pc.get_xyz), M=len(original_pc.get_xyz), requested_K=len(original_pc.get_xyz))
        cam = camera.create(frame)
        kwargs = dict(ratio_budget=.5) if method == 'ratio_50' else dict(budget=int(method[6:]))
        packet, ids, stats = engine.prepare(cam, **kwargs)
        M = stats['candidate_gaussians']
        requested = math.floor(.5*M) if method == 'ratio_50' else int(method[6:])
        assert len(ids) == min(requested, stats.get('eligible_candidates', M)), (frame, method, stats)
        return cam, packet, ids, dict(K=len(ids), M=M, requested_K=requested)

    def raster(cam, packet, method):
        if method == 'cartgs_full':
            out = p.render_prepared(cam, packet, source.scene.background, return_radii=True)
            return out['render'], out['radii']
        return render.render(cam, packet, source.scene.background)

    quality, checks = [], []
    for index, frame in enumerate(frames):
        view = source.scene.views[frame]
        truth = cv2.imread(view['image'], cv2.IMREAD_COLOR)
        assert truth is not None
        truth = cv2.cvtColor(cv2.resize(truth, SIZE, interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        gt = torch.from_numpy(truth).to('cuda').permute(2, 0, 1)/255.
        for method in METHODS:
            cam, packet, ids, stats = prepare(frame, method)
            out = counted(binding, cam, packet, source.scene.background)
            image = out[2]
            assert image.shape == (3, SIZE[1], SIZE[0]) and torch.isfinite(image).all()
            projected, contributed = int((out[3] > 0).sum()), int((out[5] > 0).sum())
            assert 0 <= contributed <= projected <= stats['K']
            psnr = float(-10*torch.log10((image.clamp(0, 1)-gt).square().mean()))
            assert math.isfinite(psnr)
            if frame in [all_frames[0], all_frames[len(all_frames)//2], all_frames[-1]]:
                normal, radii = raster(cam, packet, method)
                assert p.bits_equal(normal, image) and p.bits_equal(radii, out[3]), (frame, method)
                again = prepare(frame, method)
                if ids is not None:
                    assert torch.equal(ids, again[2])
                checks.append(dict(frame=frame, method=method, diagnostic_rgb_radii_bitwise=True,
                    independent_prepare_ids_equal=True))
            quality.append(dict(frame=frame, rgb_index=view['rgb_index'], timestamp=view['timestamp'], method=method,
                map_N=len(original_pc.get_xyz) if method == 'cartgs_full' else engine.total,
                **stats, projected_K=projected, contributing_K=contributed, psnr=psnr,
                ids_sha256='all_map' if ids is None else hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()))
            del out, image
        if index % 100 == 0 or index == len(frames)-1:
            write('QUALITY_COUNTS.csv', quality)
            print('QUALITY', index+1, len(frames), flush=True)
    write('CHECKS.csv', checks)
    for frame in np.linspace(0, len(all_frames)-1, 24 if not args.smoke else 3, dtype=int):
        for method in METHODS:
            cam, packet, ids, stats = prepare(int(frame), method)
            raster(cam, packet, method)
    rows = []
    for repeat in range(1 if args.smoke else 3):
        ordered = list(frames)
        if repeat == 1:
            ordered.reverse()
        elif repeat == 2:
            np.random.default_rng(0).shuffle(ordered)
        for index, frame in enumerate(ordered):
            for method in p.rotate(METHODS, repeat+index):
                torch.cuda.synchronize()
                tick = time.perf_counter_ns()
                cam, packet, ids, stats = prepare(frame, method)
                torch.cuda.synchronize()
                prepare_ms = (time.perf_counter_ns()-tick)/1e6
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                result = raster(cam, packet, method)
                end.record()
                end.synchronize()
                raster_ms = start.elapsed_time(end)
                torch.cuda.synchronize()
                tick = time.perf_counter_ns()
                cam, packet, ids, stats = prepare(frame, method)
                result = raster(cam, packet, method)
                torch.cuda.synchronize()
                total_ms = (time.perf_counter_ns()-tick)/1e6
                rows.append(dict(frame=frame, method=method, repeat=repeat, order=index, **stats,
                    prepare_ms=prepare_ms, raster_ms=raster_ms, total_ms=total_ms))
            if index % 200 == 0 or index == len(ordered)-1:
                write('TIMINGS.csv', rows)
                print('TIMING', repeat, index+1, len(ordered), flush=True)
    after = {name: h.sha(name) for name in before}
    changed = [name for name in before if before[name] != after[name]]
    save('PROTECTED_AFTER.json', after)
    result = dict(passed=not changed, changed=changed, protected_files=len(before), frames=len(frames),
        quality_rows=len(quality), timing_rows=len(rows), parity_checks=len(checks),
        renders_per_frame=1, render_images_saved=0, training=False, production_changed=False)
    assert render.native.stats()[1:3] == [0, 0]
    save('VALIDATION.json', result)
    assert result['passed']
    print('COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
