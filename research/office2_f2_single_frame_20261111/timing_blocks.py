"""每200帧整块按外部活动验收；受干扰块完整重测，不按性能筛选。"""
from pathlib import Path
import csv
import fcntl
import hashlib
import importlib.util
import json
import time

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('paired_protocol', ROOT/'benchmark.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
h, p, torch, np = b.h, b.p, b.torch, b.np


@torch.no_grad()
def main():
    out = ROOT/'blocks'
    out.mkdir(exist_ok=False)
    lock = (out/'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save = lambda name, obj: p.save(out/name, obj)
    write = lambda name, rows: p.write_csv(out/name, rows)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    c, d, cm, dm = h.configure()
    frozen = json.loads((b.F2_ROOT/'background_stage/reports/FROZEN.json').read_text())
    assert h.sha(frozen['binary']) == frozen['binary_sha256']
    for name, sha in frozen['sources'].items():
        assert h.sha(b.F2_ROOT/'background_stage'/name) == sha
    import sys
    sys.path.insert(0, str(b.F2_ROOT))
    spec = importlib.util.spec_from_file_location('f2_block_runtime', b.F2_ROOT/'background_runtime.py')
    f2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f2)
    f2.extension = lambda: h.verified('mycartgs_render_background_scaling_v1', Path(frozen['binary']), frozen['binary_sha256'])
    before = json.loads((ROOT/'formal/PROTECTED_AFTER.json').read_text())
    assert all(h.sha(name) == sha for name, sha in before.items())
    save('PROTECTED_BEFORE.json', before)
    assert json.loads((ROOT/'formal/VALIDATION.json').read_text())['passed']
    quality = list(csv.DictReader((ROOT/'formal/QUALITY_COUNTS.csv').open()))
    expected = {(r['method'], int(r['frame'])): r for r in quality}
    scene, source = p.load_scene(b.b.prior.SLUG)
    student, student_path, payload = p.student_for(source, b.b.prior.SLUG)
    engine = h.engine(source, student, 'D')
    camera = h.Camera(source, 'D', dm, b.b.SIZE)
    vanilla = b.b.prior.VANILLA
    ply, = list((vanilla/'model/15900_shutdown/ply/point_cloud').glob('iteration_*/point_cloud.ply'))
    pc = b.b.prior.GaussianModel(3)
    pc.load_ply(str(ply), 'cuda')
    for name in ['_xyz', '_scaling', '_rotation', '_opacity', '_features_dc', '_features_rest']:
        getattr(pc, name).requires_grad_(False)
    poses = {}
    records = {float(r[0]): r for r in np.atleast_2d(np.loadtxt(vanilla/'model/CameraTrajectory_TUM.txt'))}
    for frame, view in enumerate(source.scene.views):
        r = records[float(view['timestamp'])]
        pose = np.eye(4)
        pose[:3, :3] = b.b.Rotation.from_quat(r[4:8]).as_matrix()
        pose[:3, 3] = r[1:4]
        poses[frame] = pose
    renderers = {'G': p.create_renderer(camera.create(0), source.scene.background, engine.total, use_graph=True), 'F2': f2.Renderer()}
    methods = b.b.METHODS
    assert len(source.scene.views) == 2000 and engine.total == 43197 and len(pc.get_xyz) == 84203
    for name in ['QUALITY_COUNTS.csv', 'CHECKS.csv', 'MODEL_IDENTITIES.json']:
        (out/name).write_bytes((ROOT/'formal'/name).read_bytes())
    protocol = json.loads((ROOT/'formal/PROTOCOL.json').read_text())
    protocol.update(block_frames=200, rounds=3,
        acceptance='entire block rejected on any external GPU activity or external >10% single-process CPU sample in [start-2s,end+2s]; wait until 3 quiet samples; no latency threshold',
        quality_source=str(ROOT/'formal'), timing_reused=False)
    save('PROTOCOL.json', protocol)

    def prepare(frame, method):
        if method == 'cartgs_full':
            cam = p.MiniCam(*b.b.SIZE, camera.fx, camera.fy, np.linalg.inv(poses[frame]))
            packet = b.b.renderer.prepare_render_data(pc)
            ids = None
            stats = dict(K=84203, M=84203, requested_K=84203)
        else:
            cam = camera.create(frame)
            request = dict(ratio_budget=.5) if method == 'ratio_50' else dict(budget=int(method[6:]))
            packet, ids, st = engine.prepare(cam, **request)
            stats = dict(K=len(ids), M=st['candidate_gaussians'], requested_K=st['requested_budget'])
        assert all(stats[k] == int(expected[method, frame][k]) for k in stats)
        return cam, packet, ids, stats

    def raster(cam, packet, version):
        if version == 'original':
            return p.render_prepared(cam, packet, source.scene.background, return_radii=True)
        return renderers[version].render(cam, packet, source.scene.background)

    monitor = ROOT/'logs/blocks_MONITOR.jsonl'
    deadline = time.time()+1800

    def observations():
        complete = monitor.read_text().rsplit('\n', 1)[0]
        return [json.loads(line) for line in complete.splitlines() if line.strip()]

    def quiet():
        obs = observations()
        return len(obs) >= 3 and all(not r['foreign_active'] and not r['foreign_cpu_active'] for r in obs[-3:])

    def wait_quiet():
        while not quiet():
            assert time.time() < deadline, '等待空闲超过30分钟，保留已验收数据'
            save('STATUS.json', dict(state='waiting_external_load', accepted_blocks=len(accepted), attempts=len(attempts)))
            time.sleep(2)

    accepted, attempts, all_rows, rows = [], [], [], []
    wait_quiet()
    # 全部视角暖机并核对选点；只缓存执行计划，不保存视角网络输出。
    for frame in range(2000):
        for method in methods:
            cam, packet, ids, stats = prepare(frame, method)
            if ids is not None:
                assert hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest() == expected[method, frame]['ids_sha256']
            for version in ['original'] if method == 'cartgs_full' else ['G', 'F2']:
                raster(cam, packet, version)
    torch.cuda.synchronize()
    for repeat in range(3):
        frames = list(range(2000))
        if repeat == 1:
            frames.reverse()
        elif repeat == 2:
            np.random.default_rng(0).shuffle(frames)
        for block in range(10):
            while True:
                wait_quiet()
                current = []
                start_time = time.time()
                for index in range(block*200, (block+1)*200):
                    frame = frames[index]
                    for method in p.rotate(methods, repeat+index):
                        torch.cuda.synchronize()
                        tick = time.perf_counter_ns()
                        cam, packet, ids, stats = prepare(frame, method)
                        torch.cuda.synchronize()
                        prepare_ms = (time.perf_counter_ns()-tick)/1e6
                        versions = ['original'] if method == 'cartgs_full' else (['G', 'F2'] if (repeat+index)%2 == 0 else ['F2', 'G'])
                        for version in versions:
                            torch.cuda.synchronize()
                            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            begin.record()
                            result = raster(cam, packet, version)
                            end.record()
                            end.synchronize()
                            raster_ms = begin.elapsed_time(end)
                            torch.cuda.synchronize()
                            tick = time.perf_counter_ns()
                            current_cam, current_packet, current_ids, current_stats = prepare(frame, method)
                            result = raster(current_cam, current_packet, version)
                            torch.cuda.synchronize()
                            total_ms = (time.perf_counter_ns()-tick)/1e6
                            assert stats == current_stats
                            current.append(dict(frame=frame, method=method, version=version, repeat=repeat,
                                order=index, **stats, prepare_ms=prepare_ms, raster_ms=raster_ms, total_ms=total_ms))
                finish_time = time.time()
                while observations()[-1]['time'] < finish_time+2:
                    time.sleep(.5)
                observed = [r for r in observations() if start_time-2 <= r['time'] <= finish_time+2]
                gpu = sum(bool(r['foreign_active']) for r in observed)
                cpu = sum(bool(r['foreign_cpu_active']) for r in observed)
                passed = len(observed) >= 4 and gpu == 0 and cpu == 0
                attempt = dict(repeat=repeat, block=block, attempt=len(attempts), start_unix=start_time,
                    end_unix=finish_time, observations=len(observed), external_gpu_samples=gpu,
                    external_cpu_samples=cpu, accepted=passed, samples=len(current))
                attempts.append(attempt)
                all_rows.extend(dict(r, attempt=attempt['attempt'], accepted=passed) for r in current)
                write('ALL_TIMINGS.csv', all_rows)
                save('BLOCK_ATTEMPTS.json', attempts)
                print('BLOCK', json.dumps(attempt), flush=True)
                if passed:
                    accepted.append(attempt)
                    rows.extend(current)
                    write('TIMINGS.csv', rows)
                    break
    changed = [name for name, sha in before.items() if h.sha(name) != sha]
    save('PROTECTED_AFTER.json', {name: h.sha(name) for name in before})
    result = dict(passed=not changed and len(accepted) == 30, frames=2000, quality_rows=len(quality),
        timing_rows=len(rows), parity_checks=14000, previous_ids_K_M_counts_psnr_exact=True,
        G_F2_RGB_radii_bitwise=True, quality_source=str(ROOT/'formal'),
        accepted_blocks=len(accepted), rejected_blocks=len(attempts)-len(accepted),
        external_activity_filter_only=True, accepted_window_external_gpu_cpu_samples=0,
        protected_files=len(before), changed=changed, renders_per_frame=1,
        render_images_saved=0, training=False, production_changed=False,
        F2_stats=renderers['F2'].native.stats())
    save('VALIDATION.json', result)
    save('STATUS.json', dict(state='complete', **result))
    assert result['passed'] and len(rows) == 42000
    print('COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
