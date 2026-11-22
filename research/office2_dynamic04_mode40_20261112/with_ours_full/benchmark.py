"""补充未选择、未补偿Master Full；六方法同轮单图纯渲染复测。"""
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
PRIOR = ROOT.parent
spec = importlib.util.spec_from_file_location('full_curve_prior', PRIOR/'benchmark.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
h, p, torch, np, b = d.h, d.p, d.torch, d.np, d.b
import cv2

METHODS = ['cartgs_full', 'ours_full', 'ratio_50', 'fixed_2000', 'fixed_4000', 'dynamic_04']
SIZE = b.prior.b.SIZE


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
    save = lambda name, value: p.save(output/name, value)
    write = lambda name, rows: p.write_csv(output/name, rows)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    c, config, cm, dm = h.configure()
    before = json.loads((PRIOR/'formal/PROTECTED_AFTER.json').read_text())
    for filename in ['BUDGETS_FROZEN.csv', 'QUALITY_COUNTS.csv', 'VALIDATION.json']:
        path = PRIOR/'formal'/filename
        before[str(path)] = h.sha(path)
    assert all(h.sha(path) == sha for path, sha in before.items())
    prior_validation = json.loads((PRIOR/'formal/VALIDATION.json').read_text())
    assert prior_validation['passed']
    assert h.sha(PRIOR/'formal/BUDGETS_FROZEN.csv') == prior_validation['frozen_budget_sha256']
    save('PROTECTED_BEFORE.json', before)
    binary = b.MODE40/'build/mycartgs_footprint_ninehour_v1.so'
    binary_sha = '4de72de81cbfa1eaaee6d065ce643c57ee875be29b34c690891baf389232656e'
    spec = importlib.util.spec_from_file_location('full_curve_mode40', b.MODE40/'render_runtime.py')
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    runtime.extension = lambda: h.verified('mycartgs_footprint_ninehour_v1', binary, binary_sha)
    scene, source = p.load_scene(b.prior.b.prior.SLUG)
    student, student_path, payload = p.student_for(source, b.prior.b.prior.SLUG)
    engine = h.engine(source, student, 'D')
    camera = h.Camera(source, 'D', dm, SIZE)
    render = runtime.Renderer(40)
    master, master_ids = engine.full()
    assert engine.total == 43197 and len(master_ids) == 43197
    assert torch.equal(master_ids, torch.arange(engine.total, device=master_ids.device))
    vanilla = b.prior.b.prior.VANILLA
    ply, = list((vanilla/'model/15900_shutdown/ply/point_cloud').glob('iteration_*/point_cloud.ply'))
    pc = b.prior.b.prior.GaussianModel(3)
    pc.load_ply(str(ply), 'cuda')
    for attribute in ['_xyz', '_scaling', '_rotation', '_opacity', '_features_dc', '_features_rest']:
        getattr(pc, attribute).requires_grad_(False)
    assert len(pc.get_xyz) == 84203 and len(source.scene.views) == 2000
    records = {float(row[0]): row for row in np.atleast_2d(np.loadtxt(vanilla/'model/CameraTrajectory_TUM.txt'))}
    poses = {}
    for frame, view in enumerate(source.scene.views):
        row = records[float(view['timestamp'])]
        pose = np.eye(4)
        pose[:3, :3] = b.prior.b.Rotation.from_quat(row[4:8]).as_matrix()
        pose[:3, 3] = row[1:4]
        poses[frame] = pose
    old_quality = {(row['method'], int(row['frame'])): row for row in read(PRIOR/'formal/QUALITY_COUNTS.csv')}
    frozen = {int(row['frame']): row for row in read(PRIOR/'formal/BUDGETS_FROZEN.csv')}
    frames = [0, 500, 650, 700, 722, 741, 1000, 1500, 1900, 1999] if args.smoke else list(range(2000))
    checks = []
    save('PROTOCOL.json', dict(dataset='Replica/office2/rgbd', frames=len(frames), width=SIZE[0], height=SIZE[1],
        methods=METHODS, mode=40, renders_per_request=1, repeats=1 if args.smoke else 3,
        ours_full='all 43197 Master Gaussians; engine.full(); no Selector, Compensation, nominal Anchor culling or ratio=1 LoD request',
        quality='reuse verified old five methods; new Full actual Mode40 output vs INTER_LINEAR resized RGB',
        count='distinct Gaussians accepted in Alpha composition; Mode40 native diagnose, excluded from timing',
        timing='six methods remeasured together, CUDA Event around one raster call; camera, networks, diagnostics, GT and I/O excluded',
        validation='forward/reverse/shuffled traversal; rotated method order; reject whole 200-frame block on external activity',
        frozen_budget_sha256=prior_validation['frozen_budget_sha256'], search=False,
        production_changed=False, training=False, images_saved=0))
    save('MODEL_IDENTITIES.json', dict(scene=scene, ours_N=engine.total, vanilla_N=len(pc.get_xyz),
        student=str(student_path), student_sha256=h.sha(student_path),
        M40_binary=str(binary), M40_sha256=binary_sha, master_ids_sha256=hashlib.sha256(master_ids.cpu().numpy().tobytes()).hexdigest()))

    def prepare(frame, method):
        if method == 'cartgs_full':
            cam = p.MiniCam(*SIZE, camera.fx, camera.fy, np.linalg.inv(poses[frame]))
            return cam, b.prior.b.renderer.prepare_render_data(pc), None, dict(K=84203, M=84203, ratio=1.)
        cam = camera.create(frame)
        if method == 'ours_full':
            return cam, master, master_ids, dict(K=43197, M=43197, ratio=1.)
        request = dict(ratio_budget=.5) if method == 'ratio_50' else dict(budget=int(method[6:])) if method.startswith('fixed_') else dict(budget=int(frozen[frame]['K']))
        packet, ids, stats = engine.prepare(cam, **request)
        old = old_quality[method, frame]
        assert len(ids) == int(old['K']) and stats['candidate_gaussians'] == int(old['M'])
        return cam, packet, ids, dict(K=len(ids), M=stats['candidate_gaussians'], ratio=stats['ratio'])

    def raster(cam, packet, method):
        if method == 'cartgs_full':
            value = p.render_prepared(cam, packet, source.scene.background, return_radii=True)
            return value['render'], value['radii']
        return render.render(cam, packet, source.scene.background)

    quality = [row for row in old_quality.values() if int(row['frame']) in frames]
    for index, frame in enumerate(frames):
        cam, packet, ids, stats = prepare(frame, 'ours_full')
        image, radii = raster(cam, packet, 'ours_full')
        snapshot = image.clone()
        assert image.shape == (3, SIZE[1], SIZE[0]) and torch.isfinite(image).all()
        debug, flags, visits = render.native.diagnose()
        assert p.bits_equal(debug, snapshot), (frame, 'Full diagnostic differs')
        projected, contributed = int((radii > 0).sum()), int((flags != 0).sum())
        assert 0 <= contributed <= projected <= 43197
        truth = cv2.imread(source.scene.views[frame]['image'], cv2.IMREAD_COLOR)
        assert truth is not None
        truth = cv2.cvtColor(cv2.resize(truth, SIZE, interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        gt = torch.from_numpy(truth).to('cuda').permute(2, 0, 1)/255.
        psnr = float(-10*torch.log10((snapshot.clamp(0, 1)-gt).square().mean()))
        quality.append(dict(frame=frame, method='ours_full', **stats, requested_K=43197, eligible_N=43197,
            projected_K=projected, contributing_K=contributed, pixel_gaussian_interactions=int(visits[0]),
            psnr=psnr, ids_sha256='all_map', mode40_diagnostic_RGB_bitwise=True))
        if args.smoke or frame in [0, 500, 722, 1000, 1500, 1999]:
            for method in METHODS:
                if method == 'ours_full':
                    continue
                check_cam, check_packet, check_ids, _ = prepare(frame, method)
                actual, _ = raster(check_cam, check_packet, method)
                actual_psnr = float(-10*torch.log10((actual.clamp(0, 1)-gt).square().mean()))
                expected = old_quality[method, frame]
                assert actual_psnr == float(expected['psnr']), (frame, method, actual_psnr, expected['psnr'])
                if check_ids is not None:
                    assert hashlib.sha256(check_ids.cpu().numpy().tobytes()).hexdigest() == expected['ids_sha256']
                checks.append(dict(frame=frame, method=method, psnr_exact=True, ids_exact=True))
        if index % 100 == 0 or index == len(frames)-1:
            write('QUALITY_COUNTS.csv', quality)
            print('QUALITY', index+1, len(frames), flush=True)
    write('BASELINE_CHECKS.csv', checks)
    monitor = ROOT/'logs'/(name+'_MONITOR.jsonl')
    attempts, rows, all_rows = [], [], []
    deadline = time.time()+1800

    def observations():
        return [json.loads(line) for line in monitor.read_text().rsplit('\n', 1)[0].splitlines() if line.strip()]

    def quiet():
        while True:
            obs = observations()
            if len(obs) >= 3 and all(not row['foreign_active'] and not row['foreign_cpu_active'] for row in obs[-3:]):
                return
            assert time.time() < deadline, '等待外部任务超过30分钟，保留现场'
            save('STATUS.json', dict(state='waiting_external_load', accepted_blocks=sum(row['accepted'] for row in attempts)))
            time.sleep(2)

    quiet()
    for frame in np.linspace(0, 1999, 24 if not args.smoke else 3, dtype=int):
        for method in METHODS:
            cam, packet, ids, stats = prepare(int(frame), method)
            raster(cam, packet, method)
    torch.cuda.synchronize()
    for repeat in range(1 if args.smoke else 3):
        ordered = list(frames)
        if repeat == 1:
            ordered.reverse()
        elif repeat == 2:
            np.random.default_rng(0).shuffle(ordered)
        for block, lo in enumerate(range(0, len(ordered), 200)):
            while True:
                quiet()
                current = []
                start = time.time()
                for index in range(lo, min(lo+200, len(ordered))):
                    frame = ordered[index]
                    for method in p.rotate(METHODS, repeat+index):
                        cam, packet, ids, stats = prepare(frame, method)
                        if repeat == 0 and ids is not None and method != 'ours_full':
                            assert hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest() == old_quality[method, frame]['ids_sha256']
                        torch.cuda.synchronize()
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        tick = time.perf_counter_ns()
                        begin.record()
                        result = raster(cam, packet, method)
                        end.record()
                        end.synchronize()
                        current.append(dict(frame=frame, method=method, repeat=repeat, order=index, **stats,
                            raster_ms=begin.elapsed_time(end), raster_wall_ms=(time.perf_counter_ns()-tick)/1e6))
                finish = time.time()
                while observations()[-1]['time'] < finish+2:
                    time.sleep(.5)
                obs = [row for row in observations() if start-2 <= row['time'] <= finish+2]
                gpu, cpu = sum(bool(row['foreign_active']) for row in obs), sum(bool(row['foreign_cpu_active']) for row in obs)
                passed = len(obs) >= 3 and gpu == cpu == 0
                attempt = dict(repeat=repeat, block=block, attempt=len(attempts), start_unix=start, end_unix=finish,
                    external_gpu_samples=gpu, external_cpu_samples=cpu, observations=len(obs), accepted=passed, samples=len(current))
                attempts.append(attempt)
                all_rows.extend(dict(row, attempt=attempt['attempt'], accepted=passed) for row in current)
                write('ALL_TIMINGS.csv', all_rows)
                save('BLOCK_ATTEMPTS.json', attempts)
                print('BLOCK', json.dumps(attempt), flush=True)
                if passed:
                    rows.extend(current)
                    write('TIMINGS.csv', rows)
                    break
    after = {path: h.sha(path) for path in before}
    changed = [path for path in before if before[path] != after[path]]
    save('PROTECTED_AFTER.json', after)
    result = dict(passed=not changed, changed=changed, protected_files=len(before), frames=len(frames),
        quality_rows=len(quality), timing_rows=len(rows), methods=METHODS, full_diagnostic_RGB_bitwise=True,
        baseline_quality_reused=True, baseline_check_rows=len(checks), baseline_ids_verified_all_frames=True,
        accepted_blocks=sum(row['accepted'] for row in attempts), rejected_blocks=sum(not row['accepted'] for row in attempts),
        frozen_budget_sha256=h.sha(PRIOR/'formal/BUDGETS_FROZEN.csv'), pure_render_only=True,
        production_changed=False, training=False, images_saved=0, renderer_stats=render.native.stats())
    save('VALIDATION.json', result)
    save('STATUS.json', dict(state='complete', **result))
    assert result['passed']
    print('COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
