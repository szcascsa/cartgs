"""office2高分辨率纯渲染：同一Tensor D输入，F2与mode40配对，原CaRtGS对照。"""
from pathlib import Path
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent/'office2_f2_single_frame_20261111'
MODE40 = ROOT.parent/'low_ratio_footprint_tc_9h_20261112_0011'
spec = importlib.util.spec_from_file_location('previous_f2_benchmark', PREVIOUS/'benchmark.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
h, p, torch, np = prior.h, prior.p, prior.torch, prior.np
import cv2


def read(path):
    with path.open() as stream:
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
    save = lambda name, value: p.save(output/name, value)
    write = lambda name, rows: p.write_csv(output/name, rows)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    c, d, cm, dm = h.configure()
    before = json.loads((PREVIOUS/'blocks/PROTECTED_AFTER.json').read_text())
    assert all(h.sha(path) == sha for path, sha in before.items())
    binary = MODE40/'build/mycartgs_footprint_ninehour_v1.so'
    binary_sha = '4de72de81cbfa1eaaee6d065ce643c57ee875be29b34c690891baf389232656e'
    assert h.sha(binary) == binary_sha
    paths = [binary, MODE40/'screen.py', MODE40/'render_runtime.py', MODE40/'nine_study.py']
    paths.extend(path for path in (MODE40/'source').rglob('*') if path.is_file())
    before.update({str(path): h.sha(path) for path in paths})
    save('PROTECTED_BEFORE.json', before)
    sys.path.insert(0, str(prior.F2_ROOT))
    spec = importlib.util.spec_from_file_location('paired_f2_runtime', prior.F2_ROOT/'background_runtime.py')
    f2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f2)
    frozen = json.loads((prior.F2_ROOT/'background_stage/reports/FROZEN.json').read_text())
    f2.extension = lambda: h.verified('mycartgs_render_background_scaling_v1', Path(frozen['binary']), frozen['binary_sha256'])
    spec = importlib.util.spec_from_file_location('mode40_runtime', MODE40/'render_runtime.py')
    new = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(new)
    new.extension = lambda: h.verified('mycartgs_footprint_ninehour_v1', binary, binary_sha)
    scene, source = p.load_scene(prior.b.prior.SLUG)
    student, student_path, payload = p.student_for(source, prior.b.prior.SLUG)
    engine = h.engine(source, student, 'D')
    camera = h.Camera(source, 'D', dm, prior.b.SIZE)
    vanilla = prior.b.prior.VANILLA
    ply, = list((vanilla/'model/15900_shutdown/ply/point_cloud').glob('iteration_*/point_cloud.ply'))
    pc = prior.b.prior.GaussianModel(3)
    pc.load_ply(str(ply), 'cuda')
    for attribute in ['_xyz', '_scaling', '_rotation', '_opacity', '_features_dc', '_features_rest']:
        getattr(pc, attribute).requires_grad_(False)
    records = {float(row[0]): row for row in np.atleast_2d(np.loadtxt(vanilla/'model/CameraTrajectory_TUM.txt'))}
    poses = {}
    for frame, view in enumerate(source.scene.views):
        record = records[float(view['timestamp'])]
        pose = np.eye(4)
        pose[:3, :3] = prior.b.Rotation.from_quat(record[4:8]).as_matrix()
        pose[:3, 3] = record[1:4]
        poses[frame] = pose
    assert len(source.scene.views) == 2000 and engine.total == 43197 and len(pc.get_xyz) == 84203
    methods = prior.b.METHODS
    expected = {(row['method'], int(row['frame'])): row
                for row in read(PREVIOUS/'blocks/QUALITY_COUNTS.csv')}
    frames = [0, 500, 722, 741, 1000, 1999] if args.smoke else list(range(2000))
    renderers = {'F2': f2.Renderer(), 'M40': new.Renderer(40)}
    save('PROTOCOL.json', dict(dataset='Replica/office2/rgbd', frames=len(frames), width=2064, height=2208,
        renders_per_request=1, rounds=1 if args.smoke else 3, methods=methods,
        ours_network='same 3534-parameter distilled Compensator, Tensor Core D and new preparation',
        versions=['original', 'F2', 'M40'], mode40='tile_safe_fast: folded Tile expression, __expf, FMA; original 1e-4 transmittance cutoff, approximate RGB',
        timing='single actual raster call: CUDA Event and synchronized host wall measured together; cameras, selection, networks, GT and output processing excluded',
        cold_plans='all-frame F2/M40 quality traversal then current-view warmup before timing',
        acceptance='200-frame block, external GPU or >10% single-process CPU activity in start-2/end+2s rejects whole block; no latency thresholds',
        selected_ids='every frame exact match of prior SHA and K/M; same packet F2/M40',
        quality='new RGB PSNR against INTER_LINEAR resized original RGB; radii bitwise, max/mean RGB absolute error against F2',
        point_count='K means submitted count; previous F2 contributing counts are reference only and not claimed exact for approximate M40',
        device=torch.cuda.get_device_name(), training=False, production_changed=False, render_images_saved=0))
    save('MODEL_IDENTITIES.json', dict(scene=scene, vanilla_checkpoint=str(vanilla), vanilla_ply=str(ply),
        vanilla_N=84203, ours_N=43197, student=str(student_path), student_sha256=h.sha(student_path),
        D_binary=d['binary'], D_sha256=d['binary_sha256'], F2_binary=frozen['binary'], F2_sha256=frozen['binary_sha256'],
        M40_binary=str(binary), M40_sha256=binary_sha, M40_mode=40,
        original_rasterizer_sha256=prior.b.prior.ORIGINAL_SHA))

    def prepare(frame, method):
        if method == 'cartgs_full':
            cam = p.MiniCam(*prior.b.SIZE, camera.fx, camera.fy, np.linalg.inv(poses[frame]))
            packet = prior.b.renderer.prepare_render_data(pc)
            ids = None
            stats = dict(K=84203, M=84203, requested_K=84203)
        else:
            cam = camera.create(frame)
            request = dict(ratio_budget=.5) if method == 'ratio_50' else dict(budget=int(method[6:]))
            packet, ids, st = engine.prepare(cam, **request)
            stats = dict(K=len(ids), M=st['candidate_gaussians'], requested_K=st['requested_budget'])
        assert all(stats[key] == int(expected[method, frame][key]) for key in stats)
        return cam, packet, ids, stats

    def raster(cam, packet, version):
        if version == 'original':
            out = p.render_prepared(cam, packet, source.scene.background, return_radii=True)
            return out['render'], out['radii']
        return renderers[version].render(cam, packet, source.scene.background)

    quality = []
    for index, frame in enumerate(frames):
        truth = cv2.imread(source.scene.views[frame]['image'], cv2.IMREAD_COLOR)
        assert truth is not None
        truth = cv2.cvtColor(cv2.resize(truth, prior.b.SIZE, interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        gt = torch.from_numpy(truth).to('cuda').permute(2, 0, 1)/255.
        for method in methods:
            cam, packet, ids, stats = prepare(frame, method)
            if ids is not None:
                digest = hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
                assert digest == expected[method, frame]['ids_sha256']
            original, radii = raster(cam, packet, 'original' if method == 'cartgs_full' else 'F2')
            psnr = float(-10*torch.log10((original.clamp(0, 1)-gt).square().mean()))
            assert psnr == float(expected[method, frame]['psnr'])
            if method != 'cartgs_full':
                image, new_radii = raster(cam, packet, 'M40')
                assert torch.equal(radii, new_radii) and torch.isfinite(image).all()
                error = (image-original).abs()
                current_psnr = float(-10*torch.log10((image.clamp(0, 1)-gt).square().mean()))
                row = dict(frame=frame, method=method, **stats, F2_psnr=psnr, M40_psnr=current_psnr,
                    delta_psnr=current_psnr-psnr, rgb_abs_max=float(error.max()), rgb_abs_mean=float(error.mean()),
                    radii_bitwise=True, ids_match=True, rgb_bitwise=p.bits_equal(original, image))
            else:
                row = dict(frame=frame, method=method, **stats, F2_psnr=psnr, M40_psnr=psnr, delta_psnr=0.,
                    rgb_abs_max=0., rgb_abs_mean=0., radii_bitwise=True, ids_match=True, rgb_bitwise=True)
            quality.append(row)
        if index % 100 == 0 or index == len(frames)-1:
            write('QUALITY.csv', quality)
            print('QUALITY', index+1, len(frames), flush=True)

    monitor = ROOT/'logs'/(name+'_MONITOR.jsonl')
    attempts, accepted, all_rows, rows = [], [], [], []
    deadline = time.time()+1800

    def observations():
        return [json.loads(line) for line in monitor.read_text().rsplit('\n', 1)[0].splitlines() if line.strip()]

    def wait_quiet():
        while True:
            obs = observations()
            if len(obs) >= 3 and all(not r['foreign_active'] and not r['foreign_cpu_active'] for r in obs[-3:]):
                return
            assert time.time() < deadline, '等待空闲超过30分钟，保留现场'
            save('STATUS.json', dict(state='waiting_external_load', accepted_blocks=len(accepted), attempts=len(attempts)))
            time.sleep(2)

    wait_quiet()
    for frame in np.linspace(0, 1999, 24, dtype=int):
        for method in methods:
            cam, packet, ids, stats = prepare(int(frame), method)
            for version in ['original'] if method == 'cartgs_full' else ['F2', 'M40']:
                raster(cam, packet, version)
    torch.cuda.synchronize()
    for repeat in range(1 if args.smoke else 3):
        ordered = list(frames)
        if repeat == 1:
            ordered.reverse()
        elif repeat == 2:
            np.random.default_rng(0).shuffle(ordered)
        for block, lo in enumerate(range(0, len(ordered), 200)):
            while True:
                wait_quiet()
                current = []
                begin_time = time.time()
                for index in range(lo, min(lo+200, len(ordered))):
                    frame = ordered[index]
                    for method in p.rotate(methods, repeat+index):
                        cam, packet, ids, stats = prepare(frame, method)
                        versions = ['original'] if method == 'cartgs_full' else (['F2', 'M40'] if (repeat+index)%2 == 0 else ['M40', 'F2'])
                        for version in versions:
                            torch.cuda.synchronize()
                            a, z = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            tick = time.perf_counter_ns()
                            a.record()
                            result = raster(cam, packet, version)
                            z.record()
                            z.synchronize()
                            wall_ms = (time.perf_counter_ns()-tick)/1e6
                            current.append(dict(frame=frame, method=method, version=version, repeat=repeat, order=index,
                                **stats, raster_ms=a.elapsed_time(z), raster_wall_ms=wall_ms))
                end_time = time.time()
                while observations()[-1]['time'] < end_time+2:
                    time.sleep(.5)
                obs = [r for r in observations() if begin_time-2 <= r['time'] <= end_time+2]
                gpu, cpu = sum(bool(r['foreign_active']) for r in obs), sum(bool(r['foreign_cpu_active']) for r in obs)
                passed = len(obs) >= 3 and gpu == cpu == 0
                attempt = dict(repeat=repeat, block=block, attempt=len(attempts), start_unix=begin_time, end_unix=end_time,
                    external_gpu_samples=gpu, external_cpu_samples=cpu, observations=len(obs), accepted=passed, samples=len(current))
                attempts.append(attempt)
                all_rows.extend(dict(row, attempt=attempt['attempt'], accepted=passed) for row in current)
                write('ALL_TIMINGS.csv', all_rows)
                save('BLOCK_ATTEMPTS.json', attempts)
                print('BLOCK', json.dumps(attempt), flush=True)
                if passed:
                    accepted.append(attempt)
                    rows.extend(current)
                    write('TIMINGS.csv', rows)
                    break
    after = {path: h.sha(path) for path in before}
    changed = [path for path in before if before[path] != after[path]]
    save('PROTECTED_AFTER.json', after)
    result = dict(passed=not changed, changed=changed, protected_files=len(before), frames=len(frames),
        quality_rows=len(quality), timing_rows=len(rows), radii_bitwise_all=True, ids_K_M_match_all=True,
        rgb_approximate=True, accepted_blocks=len(accepted), rejected_blocks=len(attempts)-len(accepted),
        accepted_external_gpu_cpu_samples=0, pure_raster_only=True, renders_per_request=1,
        production_changed=False, training=False, render_images_saved=0, F2_stats=renderers['F2'].native.stats(), M40_stats=renderers['M40'].native.stats())
    save('VALIDATION.json', result)
    save('STATUS.json', dict(state='complete', **result))
    assert result['passed']
    print('COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
