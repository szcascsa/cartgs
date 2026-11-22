"""六个训练ratio与未补偿Master Full的同轮、逐视角Mode40纯渲染计时。"""
from pathlib import Path
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import re
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent/'cartgs_mode40_office2_20261113'
spec = importlib.util.spec_from_file_location('trained_ratio_prior', PRIOR/'benchmark.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
h, p, torch, np, b = previous.h, previous.p, previous.torch, previous.np, previous.b
RATIOS = [.01, .05, .1, .3, .5, .7]
METHOD_RATIO = {f'ratio_{round(r*100):02d}': r for r in RATIOS}
METHODS = ['ours_full', *METHOD_RATIO]
SIZE = (2064, 2208)


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
        lines = monitor.read_text().rsplit('\n', 1)[0].splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def quiet():
        while True:
            obs = observations()
            if len(obs) >= 3 and all(not x['foreign_active'] and not x['foreign_cpu_active'] for x in obs[-3:]):
                return
            assert time.time() < deadline, '外部负载等待超过60分钟，保留现场'
            save('STATUS.json', dict(state='waiting_external_load', accepted_blocks=sum(x['accepted'] for x in attempts)))
            time.sleep(2)

    # 不在无关训练期间加载模型、生成补偿或渲染。
    quiet()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    _, config, _, dm = h.configure()
    binary = b.MODE40/'build/mycartgs_footprint_ninehour_v1.so'
    binary_sha = '4de72de81cbfa1eaaee6d065ce643c57ee875be29b34c690891baf389232656e'
    assert h.sha(binary) == binary_sha
    spec = importlib.util.spec_from_file_location('trained_ratio_mode40', b.MODE40/'render_runtime.py')
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    runtime.extension = lambda: h.verified('mycartgs_footprint_ninehour_v1', binary, binary_sha)
    scene, source = p.load_scene('Replica__office2__rgbd')
    student, student_path, _ = p.student_for(source, 'Replica__office2__rgbd')
    assert h.sha(student_path) == 'fd85285c6bd5871f4b41c85690597ccadc965d5a5ec6f474a983e4e09f547a75'
    selector_config = Path(scene['checkpoint'])/'selector_config.yaml'
    match = re.search(r'Selector\.target_ratios:\s*\[([^\]]+)\]', selector_config.read_text())
    trained = [float(value) for value in match.group(1).split(',')]
    assert np.allclose(trained, RATIOS, rtol=1e-6, atol=1e-8)
    engine = h.engine(source, student, 'D')
    camera = h.Camera(source, 'D', dm, SIZE)
    master, all_ids = engine.full()
    assert engine.total == 43197 and len(source.scene.views) == 2000
    renderer = runtime.Renderer(40)
    old = json.loads((PRIOR/'formal/PROTECTED_AFTER.json').read_text())
    before = {filename: h.sha(filename) for filename in old if Path(filename).is_file()}
    for path in [*Path(scene['checkpoint']).rglob('*'), binary, Path(student_path),
                 Path(scene['trajectory']), Path(config['binary']), Path(__file__), ROOT/'job_runner.py']:
        if path.is_file():
            before[str(path)] = h.sha(path)
    save('PROTECTED_BEFORE.json', before)
    save('MODEL_IDENTITIES.json', dict(scene=scene, ours_N=engine.total,
        fingerprint=source.fingerprint, student=str(student_path), student_sha256=h.sha(student_path),
        Mode40_binary=str(binary), Mode40_sha256=binary_sha,
        D_binary=config['binary'], D_sha256=config['binary_sha256'], checkpoint_training_ratios=trained))
    frames = [0, 500, 650, 700, 722, 741, 1000, 1500, 1900, 1999] if args.smoke else list(range(2000))
    validation_frames = frames if args.smoke else list(map(int, np.linspace(0, 1999, 24)))
    repeats = 1 if args.smoke else 3
    save('PROTOCOL.json', dict(dataset='Replica/office2/rgbd', width=SIZE[0], height=SIZE[1],
        frames=len(frames), methods=METHODS, training_ratios=RATIOS, repeats=repeats, renders_per_request=1,
        inference='same frozen checkpoint, distilled student, Tensor D and new preparation',
        budget='ratio is fixed; K=min(floor(ratio*nominal_Anchor_candidate_M), eligible_candidates)',
        full='unselected uncompensated 43197-Gaussian Master, same archived poses',
        renderer='Mode40 shared bounded instance for all methods, warmed at Full capacity',
        timing='one raster call CUDA Event; preparation/networks/diagnose/counts/GT/I/O excluded',
        speedup='per view: median(Full CUDA Event ms)/median(ratio CUDA Event ms), same 3 rounds',
        acceptance='whole 200-frame blocks rejected for external GPU or >10% CPU process activity in +/-2s window',
        orders='forward/reverse/seed0-shuffled; per-view method order rotates', smoothing=False,
        gpu=torch.cuda.get_device_name(), training=False, production_changed=False, saved_render_images=0))

    def prepare(frame, method):
        cam = camera.create(frame)
        if method == 'ours_full':
            return cam, master, all_ids, dict(N=engine.total, M=engine.total, eligible=engine.total,
                K=engine.total, ratio=1.)
        ratio = METHOD_RATIO[method]
        packet, ids, stats = engine.prepare(cam, ratio_budget=ratio)
        m, eligible = stats['candidate_gaussians'], stats['eligible_candidates']
        assert len(ids) == min(int(np.floor(ratio*m)), eligible), (frame, method, stats)
        return cam, packet, ids, dict(N=engine.total, M=m, eligible=eligible, K=len(ids), ratio=ratio)

    def raster(cam, packet):
        return renderer.render(cam, packet, source.scene.background)

    expected_full = {int(row['frame']): row for row in read(PRIOR/'formal/QUALITY_COUNTS.csv')
                     if row['method'] == 'ours_full'}
    checks, identities = [], {}
    for frame in validation_frames:
        quiet()
        for method in METHODS:
            cam, packet, ids, stats = prepare(frame, method)
            image, radii = raster(cam, packet)
            image, radii = image.clone(), radii.clone()
            debug, flags, _ = renderer.native.diagnose()
            assert p.bits_equal(debug, image) and torch.isfinite(image).all(), (frame, method)
            projected, contributed = int((radii > 0).sum()), int((flags != 0).sum())
            assert 0 <= contributed <= projected <= stats['K']
            if method == 'ours_full':
                assert projected == int(expected_full[frame]['projected_K'])
                assert contributed == int(expected_full[frame]['contributing_K'])
            checks.append(dict(frame=frame, method=method, **stats, projected_K=projected,
                contributing_K=contributed, diagnostic_RGB_bitwise=True))
    write('PACKET_CHECKS.csv', checks)
    torch.cuda.synchronize()
    save('STATUS.json', dict(state='measuring', checked_packets=len(checks)))
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
                    for method in p.rotate(METHODS, repeat+index):
                        cam, packet, ids, stats = prepare(frame, method)
                        digest = hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
                        key = (frame, method)
                        identity = (stats['M'], stats['K'], digest)
                        assert key not in identities or identities[key] == identity, (key, '选点或预算变化')
                        identities[key] = identity
                        torch.cuda.synchronize()
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        tick = time.perf_counter_ns()
                        begin.record()
                        result = raster(cam, packet)
                        end.record()
                        end.synchronize()
                        wall_ms = (time.perf_counter_ns()-tick)/1e6
                        ms = begin.elapsed_time(end)
                        assert np.isfinite(ms) and ms > 0
                        current.append(dict(frame=frame, method=method, repeat=repeat, **stats,
                            raster_ms=ms, raster_wall_ms=wall_ms,
                            ids_sha256=digest))
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
        checked_packets=len(checks), diagnostic_RGB_bitwise=True, Full_counts_match_previous=True,
        ids_and_K_stable=True, accepted_blocks=sum(x['accepted'] for x in attempts),
        rejected_blocks=sum(not x['accepted'] for x in attempts), renderer_stats=renderer.native.stats(),
        training=False, production_changed=False, saved_render_images=0)
    assert len(timings) == len(frames)*len(METHODS)*repeats
    save('VALIDATION.json', result)
    save('STATUS.json', dict(state='complete', **result))
    assert result['passed'], result
    print('COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
