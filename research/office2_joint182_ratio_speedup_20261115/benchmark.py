"""mode182：同一冻结Student、原始分辨率、逐视角Full配对计时。"""
from pathlib import Path
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import math
import re
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent/'office2_trained_ratios_native1200_speedup_20261114'
spec = importlib.util.spec_from_file_location('joint182_previous', PRIOR/'benchmark.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
h, p, torch, np = prior.h, prior.p, prior.torch, prior.np
import cv2

RELEASE = ROOT.parent/'low_ratio_footprint_tc_9h_20261112_0011/student_resume_20261114_2250'
sys.path.insert(0, str(RELEASE))
import joint_release_runtime as release

RATIOS = [.01, .03, .05, .08, .1, .2, .3, .4, .5, .6, .7]
METHOD_RATIO = {f'ratio_{round(r*100):02d}': r for r in RATIOS}
METHODS = ['ours_full', *METHOD_RATIO]
SIZE = (1200, 680)
SETTINGS = dict(mode=182, reuse=True, fused=True, finish=True)


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    name = 'smoke' if args.smoke else 'formal'
    output = ROOT/name
    output.mkdir(exist_ok=False)
    lock = (output/'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save = lambda key, value: p.save(output/key, value)
    write = lambda key, value: p.write_csv(output/key, value)
    monitor = ROOT/'logs'/(name+'_MONITOR.jsonl')
    deadline = time.time()+3600
    attempts, timings, all_timings = [], [], []

    def observations():
        if not monitor.exists():
            return []
        return [json.loads(line) for line in monitor.read_text().rsplit('\n', 1)[0].splitlines() if line.strip()]

    def quiet():
        while True:
            obs = observations()
            if len(obs) >= 3 and all(not x['foreign_active'] and not x['foreign_cpu_active'] for x in obs[-3:]):
                return
            assert time.time() < deadline, '外部负载等待超过60分钟，保留现场'
            save('STATUS.json', dict(state='waiting_external_load', accepted_blocks=sum(x['accepted'] for x in attempts)))
            time.sleep(2)

    quiet()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    h.configure()
    binary = RELEASE/'build_joint_release/mycartgs_student_joint_release_20261114_v1.so'
    binary_sha = '84ce1c9f3d4dff4fe2a03af1f26ab85c'+'d8b9fd4544277dcb84b601e0dff9874c'
    native = h.verified('mycartgs_student_joint_release_20261114_v1', binary, binary_sha)
    release.extension = lambda: native
    scene, source = p.load_scene('Replica__office2__rgbd')
    student, student_path, _ = p.student_for(source, 'Replica__office2__rgbd')
    published = json.loads((RELEASE/'reports/FINAL_STUDENT_MANIFEST.json').read_text())
    identity = next(row for row in published if row['scene'] == scene['id'])
    assert h.sha(student_path) == identity['student_sha256']
    assert source.fingerprint == identity['map_fingerprint']
    for filename, expected in identity['checkpoint_sha256'].items():
        assert h.sha(filename) == expected, filename
    native_shapes = {}
    assert (source.scene.calibration['width'], source.scene.calibration['height']) == SIZE
    for frame in [0, len(source.scene.views)-1]:
        image = cv2.imread(source.scene.views[frame]['image'], cv2.IMREAD_COLOR)
        assert image is not None and image.shape[:2] == (SIZE[1], SIZE[0])
        native_shapes[str(frame)] = list(image.shape)
    save('NATIVE_IMAGE_SHAPES.json', native_shapes)
    selector_config = Path(scene['checkpoint'])/'selector_config.yaml'
    match = re.search(r'Selector\.target_ratios:\s*\[([^\]]+)\]', selector_config.read_text())
    trained = [float(value) for value in match.group(1).split(',')]
    engine = release.JointReleaseRenderer(source, student, **SETTINGS)
    camera_spec = importlib.util.spec_from_file_location('joint182_camera', RELEASE/'source_joint_release/preparation_camera.py')
    camera_api = importlib.util.module_from_spec(camera_spec)
    camera_spec.loader.exec_module(camera_api)
    camera = camera_api.NativeCameraPreparer(p.MiniCam, native)
    master, all_ids = engine.network.full()
    assert engine.network.total == identity['N'] == 43197 and len(source.scene.views) == 2000
    background = source.scene.background
    engine.precompile(camera.from_source(source, 0), background)
    old = json.loads((PRIOR/'formal/PROTECTED_AFTER.json').read_text())
    before = {filename: h.sha(filename) for filename in old if Path(filename).is_file()}
    protected = [*Path(scene['checkpoint']).rglob('*'), *RELEASE.glob('*runtime.py'),
        *(RELEASE/'source_joint_release').rglob('*'), binary, Path(student_path),
        Path(scene['trajectory']), RELEASE/'reports/FINAL_STUDENT_MANIFEST.json', Path(__file__), ROOT/'job_runner.py']
    for path in protected:
        if path.is_file():
            before[str(path)] = h.sha(path)
    save('PROTECTED_BEFORE.json', before)
    save('MODEL_IDENTITIES.json', dict(scene=scene, ours_N=engine.network.total,
        fingerprint=source.fingerprint, student=str(student_path), student_sha256=h.sha(student_path),
        release_binary=str(binary), release_binary_sha256=binary_sha, settings=SETTINGS,
        checkpoint_training_ratios=trained, requested_ratios=RATIOS, published_identity=identity))
    frames = [0, 65, 203, 500, 722, 741, 1000, 1500, 1900, 1999] if args.smoke else list(range(2000))
    validation_frames = frames if args.smoke else list(map(int, np.linspace(0, 1999, 24)))
    repeats = 1 if args.smoke else 3
    save('PROTOCOL.json', dict(dataset='Replica/office2/rgbd', width=SIZE[0], height=SIZE[1],
        frames=len(frames), methods=METHODS, requested_ratios=RATIOS, checkpoint_training_ratios=trained,
        repeats=repeats, renders_per_request=1, native_resolution=True, resizing=False,
        inference='published frozen Student, JointReleaseRenderer with release recommended settings', settings=SETTINGS,
        budget='fixed requested local ratio; K=min(floor(ratio*nominal_Anchor_candidate_M), eligible_candidates)',
        full='unselected uncompensated 43197-Gaussian Master; same archived pose per view',
        renderer='same JointReleaseRenderer native PendingRasterizer binary, pixel_mode=200 (18+182), members=15',
        pure_timing='one native.render CUDA Event after packet/camera preparation; excludes network and preparation',
        joint_timing='render_from_source on current CPU pose, including preparation, selection, compensation, render and finite checks; synchronized host wall and CUDA Event',
        speedup='per view: median(Full ms)/median(ratio ms), same 3 rounds; pure and joint reported separately',
        acceptance='whole 200-frame blocks rejected for external GPU or >10% CPU process activity in +/-2s window',
        orders='forward/reverse/seed0-shuffled; per-view method and pure/joint order rotate',
        gpu=torch.cuda.get_device_name(), training=False, production_changed=False, saved_render_images=0))

    def condition(method):
        return 'full' if method == 'ours_full' else f'local{METHOD_RATIO[method]:g}'

    def prepare(frame, method):
        cam = camera.from_source(source, frame)
        if method == 'ours_full':
            return cam, master, all_ids, dict(N=engine.network.total, M=engine.network.total,
                eligible=engine.network.total, K=engine.network.total, ratio=1.)
        ratio = METHOD_RATIO[method]
        packet, ids, stats = engine.network.prepare(cam, ratio_budget=ratio)
        m, eligible = stats['candidate_gaussians'], stats['eligible_candidates']
        assert len(ids) == min(int(np.floor(ratio*m)), eligible), (frame, method, stats)
        return cam, packet, ids, dict(N=engine.network.total, M=m, eligible=eligible, K=len(ids), ratio=ratio)

    def raster(cam, packet):
        x, empty = packet['inputs'], background.new_empty(0)
        return engine.native.render(background, x['means3D'], empty, x['opacities'], x['scales'], x['rotations'],
            1., empty, cam.world_view_transform, cam.full_proj_transform, math.tan(cam.FoVx*.5),
            math.tan(cam.FoVy*.5), cam.image_height, cam.image_width, x['dc'], x['shs'],
            packet['sh_degree'], cam.camera_center, 5, 256, 32, False)

    def joint(frame, method):
        return engine.render_from_source(source, frame, background, condition(method), camera)

    checks, identities = [], {}
    for frame in validation_frames:
        quiet()
        for method in METHODS:
            cam, packet, ids, stats = prepare(frame, method)
            image, radii = raster(cam, packet)
            image, radii, ids = image.clone(), radii.clone(), ids.clone()
            assert image.shape == (3, SIZE[1], SIZE[0]) and torch.isfinite(image).all()
            actual, actual_radii, actual_ids, actual_stats = joint(frame, method)
            assert p.bits_equal(actual, image), (frame, method, 'joint/split RGB mismatch')
            assert p.bits_equal(actual_radii, radii), (frame, method, 'joint/split radii mismatch')
            assert torch.equal(actual_ids, ids), (frame, method, 'joint/split IDs mismatch')
            if method != 'ours_full':
                assert actual_stats['candidate_gaussians'] == stats['M']
            checks.append(dict(frame=frame, method=method, **stats, projected_K=int((radii > 0).sum()),
                split_joint_RGB_bitwise=True, split_joint_radii_bitwise=True, split_joint_ids_bitwise=True))
    write('PACKET_CHECKS.csv', checks)
    torch.cuda.synchronize()
    save('STATUS.json', dict(state='measuring', checked_packets=len(checks)))

    def event(function):
        torch.cuda.synchronize()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        tick = time.perf_counter_ns()
        begin.record()
        value = function()
        end.record()
        end.synchronize()
        wall = (time.perf_counter_ns()-tick)/1e6
        ms = begin.elapsed_time(end)
        assert np.isfinite(ms) and ms > 0
        return value, ms, wall

    for repeat in range(repeats):
        order = list(frames)
        if repeat == 1:
            order.reverse()
        if repeat == 2:
            np.random.default_rng(0).shuffle(order)
        for block, lo in enumerate(range(0, len(order), 200)):
            while True:
                quiet()
                current, start = [], time.time()
                for index in range(lo, min(lo+200, len(order))):
                    frame = order[index]
                    for offset, method in enumerate(p.rotate(METHODS, repeat+index)):
                        cam, packet, ids, stats = prepare(frame, method)
                        digest = hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
                        key, identity_key = (frame, method), (stats['M'], stats['K'], digest)
                        assert key not in identities or identities[key] == identity_key, (key, '选点或预算变化')
                        identities[key] = identity_key
                        calls = dict(pure=lambda: raster(cam, packet), joint=lambda: joint(frame, method))
                        values = {}
                        for scope in p.rotate(['pure', 'joint'], repeat+index+offset):
                            values[scope] = event(calls[scope])
                        returned_ids = values['joint'][0][2]
                        assert len(returned_ids) == stats['K'] and torch.equal(ids, returned_ids)
                        current.append(dict(frame=frame, method=method, repeat=repeat, **stats,
                            raster_ms=values['pure'][1], raster_wall_ms=values['pure'][2],
                            joint_cuda_ms=values['joint'][1], joint_wall_ms=values['joint'][2], ids_sha256=digest))
                finish = time.time()
                while observations()[-1]['time'] < finish+2:
                    time.sleep(.5)
                obs = [x for x in observations() if start-2 <= x['time'] <= finish+2]
                gpu, cpu = sum(bool(x['foreign_active']) for x in obs), sum(bool(x['foreign_cpu_active']) for x in obs)
                accepted = len(obs) >= 3 and gpu == cpu == 0
                attempt = dict(repeat=repeat, block=block, attempt=len(attempts), start_unix=start,
                    end_unix=finish, samples=len(current), observations=len(obs),
                    external_gpu_samples=gpu, external_cpu_samples=cpu, accepted=accepted)
                attempts.append(attempt)
                all_timings.extend(dict(row, attempt=attempt['attempt'], accepted=accepted) for row in current)
                write('ALL_TIMINGS.csv', all_timings)
                save('BLOCK_ATTEMPTS.json', attempts)
                print('BLOCK', json.dumps(attempt), flush=True)
                if accepted:
                    timings.extend(current)
                    write('TIMINGS.csv', timings)
                    save('STATUS.json', dict(state='measuring', accepted_rows=len(timings),
                        accepted_blocks=sum(x['accepted'] for x in attempts)))
                    break
    after = {filename: h.sha(filename) for filename in before}
    changed = [filename for filename in before if before[filename] != after[filename]]
    save('PROTECTED_AFTER.json', after)
    result = dict(passed=not changed, changed=changed, protected_files=len(before),
        frames=len(frames), methods=METHODS, repeats=repeats, timing_rows=len(timings),
        checked_packets=len(checks), split_joint_RGB_radii_ids_bitwise=True, native_resolution_verified=True,
        ids_and_K_stable=True, accepted_blocks=sum(x['accepted'] for x in attempts),
        rejected_blocks=sum(not x['accepted'] for x in attempts), renderer_stats=engine.native.stats(),
        renderer_workspace_bytes=engine.native.workspace_bytes(), training=False,
        production_changed=False, saved_render_images=0)
    assert len(timings) == len(frames)*len(METHODS)*repeats
    save('VALIDATION.json', result)
    save('STATUS.json', dict(state='complete', **result))
    assert result['passed'], result
    print('COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
