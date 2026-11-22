"""Mode40逐帧离线预算搜索：目标0.6ms，固定K后独立验证，不修改网络和生产。"""
from pathlib import Path
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import math
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent/'office2_mode40_single_frame_20261112'
spec = importlib.util.spec_from_file_location('dynamic_prior', PRIOR/'benchmark.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
h, p, torch, np = b.h, b.p, b.torch, b.np
import cv2

TARGET = .6
TOLERANCE = .03
METHODS = ['cartgs_full', 'ratio_50', 'fixed_2000', 'fixed_4000', 'dynamic_06']
GRID = [.01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--feedback-from')
    args = parser.parse_args()
    run = ('feedback_smoke' if args.smoke else 'feedback') if args.feedback_from else ('smoke' if args.smoke else 'formal')
    output = ROOT/run
    output.mkdir(exist_ok=False)
    lock = (output/'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save = lambda name, value: p.save(output/name, value)
    write = lambda name, rows: p.write_csv(output/name, rows)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    c, d, cm, dm = h.configure()
    before = json.loads((PRIOR/'formal/PROTECTED_AFTER.json').read_text())
    assert all(h.sha(path) == sha for path, sha in before.items())
    save('PROTECTED_BEFORE.json', before)
    binary = b.MODE40/'build/mycartgs_footprint_ninehour_v1.so'
    binary_sha = '4de72de81cbfa1eaaee6d065ce643c57ee875be29b34c690891baf389232656e'
    spec = importlib.util.spec_from_file_location('dynamic_mode40_runtime', b.MODE40/'render_runtime.py')
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    runtime.extension = lambda: h.verified('mycartgs_footprint_ninehour_v1', binary, binary_sha)
    scene, source = p.load_scene(b.prior.b.prior.SLUG)
    student, student_path, payload = p.student_for(source, b.prior.b.prior.SLUG)
    engine = h.engine(source, student, 'D')
    camera = h.Camera(source, 'D', dm, b.prior.b.SIZE)
    render = runtime.Renderer(40)
    vanilla = b.prior.b.prior.VANILLA
    ply, = list((vanilla/'model/15900_shutdown/ply/point_cloud').glob('iteration_*/point_cloud.ply'))
    pc = b.prior.b.prior.GaussianModel(3)
    pc.load_ply(str(ply), 'cuda')
    for name in ['_xyz', '_scaling', '_rotation', '_opacity', '_features_dc', '_features_rest']:
        getattr(pc, name).requires_grad_(False)
    records = {float(row[0]): row for row in np.atleast_2d(np.loadtxt(vanilla/'model/CameraTrajectory_TUM.txt'))}
    poses = {}
    for frame, view in enumerate(source.scene.views):
        row = records[float(view['timestamp'])]
        pose = np.eye(4)
        pose[:3, :3] = b.prior.b.Rotation.from_quat(row[4:8]).as_matrix()
        pose[:3, 3] = row[1:4]
        poses[frame] = pose
    assert len(source.scene.views) == 2000 and engine.total == 43197 and len(pc.get_xyz) == 84203
    old_counts = {(r['method'], int(r['frame'])): r
                  for r in read(b.PREVIOUS/'blocks/QUALITY_COUNTS.csv')}
    old_quality = {(r['method'], int(r['frame'])): r for r in read(PRIOR/'formal/QUALITY.csv')}
    frames = [0, 500, 650, 700, 722, 741, 1000, 1500, 1900, 1999] if args.smoke else list(range(2000))
    seed_budgets, seed_timing, seed_search, seed_quality = {}, {}, {}, {}
    if args.feedback_from:
        seed = ROOT/args.feedback_from
        assert json.loads((seed/'VALIDATION.json').read_text())['passed']
        seed_budgets = {int(r['frame']): r for r in read(seed/'BUDGETS_FROZEN.csv')}
        for r in read(seed/'TIMINGS.csv'):
            if r['method'] == 'dynamic_06':
                seed_timing.setdefault(int(r['frame']), []).append(float(r['raster_ms']))
        seed_timing = {frame: float(np.median(values)) for frame, values in seed_timing.items()}
        for r in read(seed/'SEARCH_ALL.csv'):
            if r['accepted'] == 'True':
                seed_search.setdefault(int(r['frame']), {})[int(r['K'])] = float(r['median_ms'])
        seed_quality = {(r['method'], int(r['frame'])): r for r in read(seed/'QUALITY_COUNTS.csv')}
    save('PROTOCOL.json', dict(dataset='Replica/office2/rgbd', width=2064, height=2208, frames=len(frames),
        methods=METHODS, mode=40, target_ms=TARGET, target_band_ms=[TARGET-TOLERANCE, TARGET+TOLERANCE],
        objective='minimize absolute error of measured pure render median to 0.6ms; PSNR not used in search; tie favors larger K',
        ratios=GRID, additional_grid_budgets=[2000, 4000], max_local_refinements=8,
        calibration='each frame full coarse grid then measured local bracket/secant proposals; no monotonic or linear guarantee; one warmup and three single renders per candidate',
        validation='freeze all K and SHA before quality/timing; three independent repeats in forward/reverse/shuffled order',
        ratio='for dynamic fixed budget request, network ratio equals K/nominal candidate M; clamp only to eligible pool; log both counts',
        pool_limit='do not add invisible points, duplicate points or delay rendering to reach target; record below-target cases',
        timing='CUDA Event around one raster call; camera, preparation, selection, compensation, search, diagnostics, output processing excluded',
        attribution='offline per-view oracle table; no online causal controller or mobile deployment claim',
        monitor='100-frame search blocks and 200-frame validation blocks; any external GPU or >10% single-process CPU sample in start-2/end+2s rejects full block; no latency filtering',
        quality='Mode40 actual output vs INTER_LINEAR resized original RGB; actual Alpha flags from existing native diagnose(), excluded from timing',
        device=torch.cuda.get_device_name(), training=False, production_changed=False, render_images_saved=0))
    if args.feedback_from:
        save('FEEDBACK_PROTOCOL.json', dict(source=str(ROOT/args.feedback_from),
            initial_proposal='use first independent measured latency error and local coarse slope to propose K, then remeasure actual candidate',
            context='before each of three search samples render original CaRtGS, reprepare current candidate, then time candidate only',
            goal_unchanged=True, final_validation_independent=True, quality_not_used=True,
            baseline_quality_reused=True))
    save('MODEL_IDENTITIES.json', dict(scene=scene, student=str(student_path), student_sha256=h.sha(student_path),
        D_binary=d['binary'], D_sha256=d['binary_sha256'], M40_binary=str(binary), M40_sha256=binary_sha,
        vanilla_ply=str(ply), vanilla_N=84203, ours_N=43197))
    monitor = ROOT/'logs'/(run+'_MONITOR.jsonl')
    deadline = time.time()+3600
    attempts = []

    def observations():
        return [json.loads(line) for line in monitor.read_text().rsplit('\n', 1)[0].splitlines() if line.strip()]

    def quiet():
        while True:
            obs = observations()
            if len(obs) >= 3 and all(not o['foreign_active'] and not o['foreign_cpu_active'] for o in obs[-3:]):
                return
            assert time.time() < deadline, '等待外部任务超过1小时，保留现场'
            save('STATUS.json', dict(state='waiting_external_load', attempts=len(attempts)))
            time.sleep(2)

    def assess(phase, block, start, finish, samples, repeat=-1):
        while observations()[-1]['time'] < finish+2:
            time.sleep(.5)
        obs = [o for o in observations() if start-2 <= o['time'] <= finish+2]
        gpu, cpu = sum(bool(o['foreign_active']) for o in obs), sum(bool(o['foreign_cpu_active']) for o in obs)
        passed = len(obs) >= 3 and gpu == cpu == 0
        item = dict(phase=phase, block=block, repeat=repeat, attempt=len(attempts), start_unix=start,
            end_unix=finish, samples=samples, observations=len(obs), external_gpu_samples=gpu,
            external_cpu_samples=cpu, accepted=passed)
        attempts.append(item)
        save('BLOCK_ATTEMPTS.json', attempts)
        print('BLOCK', json.dumps(item), flush=True)
        return item

    def raster(cam, packet, method):
        if method == 'cartgs_full':
            result = p.render_prepared(cam, packet, source.scene.background, return_radii=True)
            return result['render'], result['radii']
        return render.render(cam, packet, source.scene.background)

    def event(cam, packet, method):
        torch.cuda.synchronize()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        tick = time.perf_counter_ns()
        begin.record()
        result = raster(cam, packet, method)
        end.record()
        end.synchronize()
        return begin.elapsed_time(end), (time.perf_counter_ns()-tick)/1e6

    search_rows, chosen = [], []
    for block, lo in enumerate(range(0, len(frames), 100)):
        while True:
            quiet()
            batch, candidate_rows = [], []
            start = time.time()
            for frame in frames[lo:lo+100]:
                cam = camera.create(frame)
                _, _, pool = engine.prepare(cam, budget=0)
                M, available = pool['candidate_gaussians'], pool['eligible_candidates']
                assert M > 0 and 0 < available <= M
                measured = {}
                if args.feedback_from:
                    context_cam = p.MiniCam(*b.prior.b.SIZE, camera.fx, camera.fy, np.linalg.inv(poses[frame]))
                    context_packet = b.prior.b.renderer.prepare_render_data(pc)

                def probe(k, phase):
                    k = int(max(1, min(available, k)))
                    if k in measured:
                        return measured[k]
                    packet, ids, stats = engine.prepare(cam, budget=k)
                    assert len(ids) == k and stats['candidate_gaussians'] == M
                    raster(cam, packet, 'dynamic_06')  # 排除当前容量的冷图计划创建。
                    samples = []
                    for _ in range(3):
                        if args.feedback_from:
                            raster(context_cam, context_packet, 'cartgs_full')
                            packet, ids, stats = engine.prepare(cam, budget=k)
                        samples.append(event(cam, packet, 'dynamic_06')[0])
                    ms = float(np.median(samples))
                    measured[k] = ms
                    candidate_rows.append(dict(frame=frame, stage=phase, K=k, M=M, eligible_N=available,
                        ratio=k/M, sample0_ms=samples[0], sample1_ms=samples[1], sample2_ms=samples[2], median_ms=ms))
                    return ms

                coarse = sorted(set([max(1, min(available, int(r*M))) for r in GRID]+[min(2000, available), min(4000, available), available]))
                if args.feedback_from:
                    seed_k = int(seed_budgets[frame]['K'])
                    previous = seed_search[frame]
                    below = [k for k in previous if k < seed_k]
                    above = [k for k in previous if k > seed_k]
                    left = max(below) if below else seed_k
                    right = min(above) if above else seed_k
                    slope = (previous[right]-previous[left])/max(1, right-left)
                    predicted = int(round(seed_k+(TARGET-seed_timing[frame])/slope)) if slope > 1e-7 else available
                    coarse = sorted(set([seed_k, max(1, min(available, predicted)), available]))
                for k in coarse:
                    probe(k, 'coarse')
                for _ in range(8):
                    best = min(measured, key=lambda k: (abs(measured[k]-TARGET), -k))
                    if args.feedback_from and max(measured.values()) < TARGET-TOLERANCE and seed_budgets[frame]['all_tested_below_band'] == 'True':
                        break
                    if abs(measured[best]-TARGET) <= .006:
                        break
                    ordered = sorted(measured)
                    intervals = [(a, z) for a, z in zip(ordered, ordered[1:]) if z-a >= 16]
                    crossing = [(a, z) for a, z in intervals if (measured[a]-TARGET)*(measured[z]-TARGET) <= 0]
                    if crossing:
                        a, z = min(crossing, key=lambda pair: (min(abs(measured[pair[0]]-TARGET), abs(measured[pair[1]]-TARGET)), -(pair[1]-pair[0])))
                        fraction = (TARGET-measured[a])/(measured[z]-measured[a]) if measured[z] != measured[a] else .5
                        fraction = max(.15, min(.85, fraction))
                        k = int(round(a+(z-a)*fraction))
                    else:
                        local = [(a, z) for a, z in intervals if best in (a, z)]
                        if not local:
                            break
                        a, z = max(local, key=lambda pair: pair[1]-pair[0])
                        k = (a+z)//2
                    if k in measured:
                        break
                    probe(k, 'refine')
                best = min(measured, key=lambda k: (abs(measured[k]-TARGET), -k))
                packet, ids, stats = engine.prepare(cam, budget=best)
                digest = hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
                batch.append(dict(frame=frame, rgb_index=source.scene.views[frame]['rgb_index'],
                    timestamp=source.scene.views[frame]['timestamp'], K=best, M=M, eligible_N=available,
                    ratio=best/M, calibration_ms=measured[best], calibration_error_ms=measured[best]-TARGET,
                    max_pool_ms=measured[available], min_tested_ms=min(measured.values()), max_tested_ms=max(measured.values()),
                    tested_budgets=len(measured), all_tested_below_band=max(measured.values()) < TARGET-TOLERANCE,
                    selected_pool_exhausted=best == available, ids_sha256=digest))
            decision = assess('search', block, start, time.time(), len(candidate_rows))
            search_rows.extend(dict(r, attempt=decision['attempt'], accepted=decision['accepted']) for r in candidate_rows)
            write('SEARCH_ALL.csv', search_rows)
            if decision['accepted']:
                chosen.extend(batch)
                write('BUDGETS_FROZEN.csv', chosen)
                print('SEARCH_PROGRESS', len(chosen), len(frames), flush=True)
                break
    frozen_sha = h.sha(output/'BUDGETS_FROZEN.csv')
    save('BUDGET_FREEZE.json', dict(sha256=frozen_sha, frames=len(chosen), target_ms=TARGET,
        frozen_before_independent_validation=True, objective_uses_PSNR=False))
    budgets = {int(r['frame']): r for r in chosen}

    def prepare(frame, method):
        if method == 'cartgs_full':
            cam = p.MiniCam(*b.prior.b.SIZE, camera.fx, camera.fy, np.linalg.inv(poses[frame]))
            return cam, b.prior.b.renderer.prepare_render_data(pc), None, dict(K=84203, M=84203, requested_K=84203, eligible_N=84203, ratio=1.)
        cam = camera.create(frame)
        kwargs = dict(ratio_budget=.5) if method == 'ratio_50' else dict(budget=int(method[6:])) if method.startswith('fixed_') else dict(budget=budgets[frame]['K'])
        packet, ids, stats = engine.prepare(cam, **kwargs)
        result = dict(K=len(ids), M=stats['candidate_gaussians'], requested_K=stats['requested_budget'],
            eligible_N=stats['eligible_candidates'], ratio=stats['ratio'])
        if method == 'dynamic_06':
            assert result['K'] == budgets[frame]['K'] and result['M'] == budgets[frame]['M']
        return cam, packet, ids, result

    quality = []
    for index, frame in enumerate(frames):
        truth = cv2.imread(source.scene.views[frame]['image'], cv2.IMREAD_COLOR)
        truth = cv2.cvtColor(cv2.resize(truth, b.prior.b.SIZE, interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        gt = torch.from_numpy(truth).to('cuda').permute(2, 0, 1)/255.
        for method in METHODS:
            if args.feedback_from and method != 'dynamic_06':
                quality.append(seed_quality[method, frame])
                continue
            cam, packet, ids, stats = prepare(frame, method)
            image, radii = raster(cam, packet, method)
            assert torch.isfinite(image).all()
            psnr = float(-10*torch.log10((image.clamp(0, 1)-gt).square().mean()))
            digest = 'all_map' if ids is None else hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
            if method == 'dynamic_06':
                assert digest == budgets[frame]['ids_sha256']
            else:
                assert digest == old_counts[method, frame]['ids_sha256']
                assert psnr == float(old_quality[method, frame]['M40_psnr'])
            projected = int((radii > 0).sum())
            if method == 'cartgs_full':
                contributed = int(old_counts[method, frame]['contributing_K'])
                interactions = -1
            else:
                debug_image, flags, visits = render.native.diagnose()
                assert p.bits_equal(debug_image, image), (frame, method, 'diagnostic render mismatch')
                contributed, interactions = int((flags != 0).sum()), int(visits[0])
                assert 0 <= contributed <= projected <= len(ids)
            quality.append(dict(frame=frame, method=method, **stats, projected_K=projected,
                contributing_K=contributed, pixel_gaussian_interactions=interactions, psnr=psnr,
                ids_sha256=digest, mode40_diagnostic_RGB_bitwise=True))
        if index % 100 == 0 or index == len(frames)-1:
            write('QUALITY_COUNTS.csv', quality)
            print('QUALITY', index+1, len(frames), flush=True)

    rows, all_rows = [], []
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
                        ms, wall = event(cam, packet, method)
                        current.append(dict(frame=frame, method=method, repeat=repeat, order=index,
                            **stats, raster_ms=ms, raster_wall_ms=wall))
                decision = assess('validation', block, start, time.time(), len(current), repeat)
                all_rows.extend(dict(r, attempt=decision['attempt'], accepted=decision['accepted']) for r in current)
                write('ALL_TIMINGS.csv', all_rows)
                if decision['accepted']:
                    rows.extend(current)
                    write('TIMINGS.csv', rows)
                    break
    assert h.sha(output/'BUDGETS_FROZEN.csv') == frozen_sha
    after = {path: h.sha(path) for path in before}
    changed = [path for path in before if before[path] != after[path]]
    save('PROTECTED_AFTER.json', after)
    result = dict(passed=not changed, changed=changed, protected_files=len(before), frames=len(frames),
        frozen_budget_sha256=frozen_sha, search_rows=len(search_rows), quality_rows=len(quality), timing_rows=len(rows),
        diagnostic_RGB_bitwise=True, baseline_ids_PSNR_exact=True, independent_validation=True,
        target_ms=TARGET, target_band_ms=[.57, .63], pure_raster_only=True,
        renders_per_request=1, offline_oracle=True, training=False, production_changed=False, render_images_saved=0,
        renderer_stats=render.native.stats())
    save('VALIDATION.json', result)
    save('STATUS.json', dict(state='complete', **result))
    assert result['passed']
    print('COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
