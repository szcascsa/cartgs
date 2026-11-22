"""固定相机、三方法、独立GPU负载进程的配对测量。"""
import argparse
import csv
import fcntl
import importlib.util
import os
from pathlib import Path
import subprocess
import time

from common import *
from control import BudgetController, DUTIES, LEVELS, STAGE_SECONDS
from activity import ActivityMonitor


FRAME = 396
METHODS = ('cartgs_full', 'ours_fixed4000', 'ours_adaptive')
QUEUE = Path('/home/prmi/zwj/realLOD/saved_experiments/dual_view_li_best_full_20261107')


def load_module(name, path, package=False):
    spec = importlib.util.spec_from_file_location(name, path,
        submodule_search_locations=[str(Path(path).parent)] if package else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def atomic_text(path, content):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(content)
    temporary.replace(path)


def queue_paused():
    state = json.loads((QUEUE / 'LOD_PIPELINE.json').read_text())
    return (QUEUE / 'HOLD').exists() and state.get('phase') in ('held', 'paused')


def wait_quiet(monitor, status, allow_background=False):
    while True:
        if monitor.error or monitor.process.poll() is not None:
            raise RuntimeError(monitor.error or 'GPU activity monitor exited while waiting')
        ready = len(monitor.samples) >= 3 if allow_background else monitor.quiet()
        if queue_paused() and ready:
            return
        status.update(state='waiting_queue_and_gpu_telemetry' if allow_background else 'waiting_queue_and_quiet_gpu',
            queue_paused=queue_paused())
        save(ROOT / 'STATUS.json', status)
        time.sleep(1)


def load_runtime():
    import deployment_runtime
    import frozen_inference_cuda
    native = load_module('mycartgs_deployment_fp32_v11', V11_BINARY)
    frozen = load_module('mycartgs_frozen_inference_v3', FROZEN_BINARY)
    deployment_runtime.deployment_extension = lambda: native
    frozen_inference_cuda.extension = lambda: frozen
    return deployment_runtime.DeploymentRenderer.from_checkpoint(
        CHECKPOINT, 0, shrinking_radix=True, compensation_tile=4,
    )


class Renderer:
    def __init__(self):
        scene = BudgetScene(CHECKPOINT, DATASET, 4000, 0, None, 'pytorch')
        self.runtime = load_runtime()
        self.camera = camera_from_view(scene.calibration, scene.views[FRAME])
        self.gt = scene.ground_truth(FRAME)
        self.bg = scene.background
        calibration, views = load_views(VANILLA, DATASET)
        assert calibration == scene.calibration
        assert views[FRAME]['timestamp'] == scene.views[FRAME]['timestamp']
        self.cart_camera = camera_from_view(calibration, views[FRAME])
        self.cart_pc = GaussianModel(3)
        ply = list((VANILLA / 'point_cloud').glob('iteration_*/point_cloud.ply'))
        assert len(ply) == 1
        self.cart_pc.load_ply(str(ply[0]))
        self.cart_raster = load_module('original_raster', ROOT / 'source/original_raster/__init__.py', True)
        self.cart_renderer = load_module('original_renderer', ROOT / 'source/original_renderer.py')
        self.cart_renderer.GaussianRasterizationSettings = self.cart_raster.GaussianRasterizationSettings
        self.cart_renderer.GaussianRasterizer = self.cart_raster.GaussianRasterizer
        assert self.runtime.total == 13298 and len(self.cart_pc.get_xyz) == 35301

    def call(self, method, budget=4000):
        torch.cuda.synchronize()
        start = time.perf_counter()
        if method == 'cartgs_full':
            prepared = start
            output = self.cart_renderer.render(self.cart_camera, self.cart_pc, self.bg)
            k = 35301
            stats = {}
        else:
            packet, ids, stats = self.runtime.prepare(self.camera, budget=budget)
            prepared = time.perf_counter()
            output = render_prepared(self.camera, packet, self.bg)
            k = len(ids)
        torch.cuda.synchronize()
        end = time.perf_counter()
        if method != 'cartgs_full':
            assert k == min(budget, stats['eligible_candidates'])
            assert stats['device_counts'] and stats['shrinking_radix']
        return dict(
            K=k, candidate_N=stats.get('candidate_gaussians', 35301),
            eligible_N=stats.get('eligible_candidates', 35301),
            network_ratio=stats.get('ratio'), requested_K=budget if method != 'cartgs_full' else 35301,
            prepare_ms=1000 * (prepared - start), render_ms=1000 * (end - prepared),
            total_ms=1000 * (end - start),
        ), output['render']

    def quality(self, folder):
        rows = []
        signatures = {}
        for method, budget in [('cartgs_full', 35301)] + [('ours', k) for k in LEVELS]:
            record, image = self.call(method, budget)
            reference = image.clone()
            _, repeated = self.call(method, budget)
            assert torch.equal(reference.view(torch.uint8), repeated.view(torch.uint8))
            assert torch.isfinite(reference).all()
            psnr = float(-10 * torch.log10((reference.clamp(0, 1) - self.gt).square().mean()))
            signature = hashlib.sha256(reference.cpu().numpy().tobytes()).hexdigest()
            signatures[method, budget] = signature
            rows.append(dict(method=method, budget=budget, K=record['K'], psnr=psnr,
                image_sha256=signature, repeated_image_bitwise=True))
        write_csv(folder / 'QUALITY.csv', rows)
        return signatures


def run_condition(renderer, monitor, folder, method, repeat, target, stage_seconds, status, allow_background=False):
    folder.mkdir()
    command, ready, stop = [folder / name for name in ('command', 'ready', 'stop')]
    atomic_text(command, '-1 0\n')
    with (folder / 'load_stderr.log').open('w') as log:
        load = subprocess.Popen([str(ROOT / 'build/gpu_load'), '--ready-file', str(ready),
            '--command-file', str(command), '--stop-file', str(stop), '--log', str(folder / 'load.csv'),
            '--parent-pid', str(os.getpid())],
            stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 20
        try:
            while not ready.exists():
                if load.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('GPU load worker failed to become ready')
                time.sleep(.02)
            for _ in range(20):
                renderer.call(method, 4000)
            time.sleep(.2)
            load_identity = dict(pid=load.pid, start_ticks=Path(f'/proc/{load.pid}/stat').read_text().split()[21],
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
            save(folder / 'LOAD_IDENTITY.json', load_identity)
            controller = BudgetController(target)
            rows = []
            boundaries = []
            start = time.monotonic()
            active_stage = -1
            period = .005
            next_request = start
            while time.monotonic() - start < len(DUTIES) * stage_seconds:
                now = time.monotonic()
                stage = min(len(DUTIES) - 1, int((now - start) // stage_seconds))
                if stage != active_stage:
                    atomic_text(command, f'{stage} {DUTIES[stage]}\n')
                    active_stage = stage
                    boundaries.append(dict(stage=stage, target_duty=DUTIES[stage], elapsed_s=now-start, monotonic_s=now))
                    status.update(state='timing', method=method, repeat=repeat, stage=stage, budget=controller.budget)
                    save(ROOT / 'STATUS.json', status)
                    if not queue_paused():
                        raise RuntimeError('Experiment queue resumed during measurement')
                budget = controller.budget if method == 'ours_adaptive' else 4000
                record, image = renderer.call(method, budget)
                elapsed = time.monotonic() - start
                if method == 'ours_adaptive':
                    controller.update(record['total_ms'])
                record.update(method=method, repeat=repeat, stage=stage, target_duty=DUTIES[stage],
                    budget=budget if method != 'cartgs_full' else 35301,
                    elapsed_s=now-start, completed_s=elapsed, monotonic_s=now,
                    crossed_stage=elapsed >= (stage+1)*stage_seconds,
                    next_budget=controller.budget if method == 'ours_adaptive' else budget,
                    tau_ms=target, over_target=record['total_ms'] > target)
                rows.append(record)
                next_request += period
                delay = next_request - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_request = time.monotonic()
            end = time.monotonic()
            atomic_text(command, '-1 0\n')
            atomic_text(stop, 'stop\n')
            load.wait(timeout=10)
            if load.returncode != 0:
                raise RuntimeError('GPU load process reported an error')
            time.sleep(4)
            samples, unexpected = monitor.unexpected(start-2, end+3, {os.getpid(), load.pid})
            coverage = len(samples) >= int(end-start)+3
            result = dict(passed=coverage and (not unexpected or allow_background), coverage=coverage,
                shared_gpu_background_allowed=allow_background,
                unexpected_gpu_activity=unexpected, pmon_samples=len(samples),
                start_monotonic_s=start, end_monotonic_s=end, rows=len(rows), stages=boundaries)
            save(folder / 'ACCEPTANCE.json', result)
            write_csv(folder / 'TIMINGS.csv', rows)
            if not result['passed']:
                raise RuntimeError('Unexpected GPU activity or missing telemetry coverage')
            return rows
        finally:
            atomic_text(stop, 'stop\n')
            if load.poll() is None:
                try:
                    load.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    load.terminate()
                    load.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--label', required=True)
    parser.add_argument('--allow-recorded-background', action='store_true')
    args = parser.parse_args()
    lock = (ROOT / 'reports/run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    folder = ROOT / 'runs' / args.label
    folder.mkdir()
    status = dict(state='waiting', process=identity(), label=args.label, smoke=args.smoke)
    background = json.loads((ROOT / 'reports/BACKGROUND_ALLOWLIST.json').read_text()) if args.allow_recorded_background else None
    if background:
        assert background['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        for item in background['processes']:
            proc = Path('/proc') / str(item['pid'])
            if proc.exists():
                assert proc.joinpath('stat').read_text().split()[21] == item['start_ticks']
        save(folder / 'AUTHORIZED_BACKGROUND.json', background)
    monitor = ActivityMonitor(folder, [item['pid'] for item in background['processes']] if background else ())
    try:
        wait_quiet(monitor, status, args.allow_recorded_background)
        protected = json.loads((ROOT / 'reports/PROTECTED_BEFORE.json').read_text())
        assert all(sha(name) == value for name, value in protected.items())
        assert sha(V11_BINARY) == 'cb41244904755c25c02a1dd601b84514c83b6982c6e862043a4eb9eca7e59899'
        with torch.inference_mode(False), torch.no_grad():
            renderer = Renderer()
        with torch.inference_mode():
            expected = renderer.quality(folder)
            calibration = []
            for _ in range(100):
                renderer.call('ours_fixed4000', 4000)
                renderer.call('cartgs_full')
            calibration_start = time.monotonic()
            for _ in range(200):
                calibration.append(renderer.call('ours_fixed4000', 4000)[0]['total_ms'])
            calibration_end = time.monotonic()
            time.sleep(4)
            _, interference = monitor.unexpected(calibration_start-2, calibration_end+3, {os.getpid()})
            if interference and not args.allow_recorded_background:
                raise RuntimeError('External GPU activity during target calibration')
            target = 1.1 * float(np.percentile(calibration, 95))
            save(folder / 'CALIBRATION.json', dict(samples_ms=calibration, p95_ms=target/1.1, tau_ms=target,
                other_background_activity=interference))
            save(folder / 'ENVIRONMENT.json', dict(frame=FRAME, own_checkpoint=str(CHECKPOINT), cartgs_checkpoint=str(VANILLA),
                binary_sha256=sha(V11_BINARY), methods=METHODS, duties=DUTIES, stage_seconds=.5 if args.smoke else STAGE_SECONDS,
                request_period_ms=5, target_ms=target, cuda=torch.version.cuda, torch=torch.__version__, gpu=gpu_state('before'),
                device_counts=True, shrinking_radix=True, compensation_tile=4, tf32=False,
                shared_gpu=bool(background and background['processes']), authorized_background=background))
            rows = []
            repeats = 1 if args.smoke else 3
            for repeat in range(repeats):
                methods = ('ours_adaptive',) if args.smoke else METHODS[repeat:] + METHODS[:repeat]
                for method in methods:
                    wait_quiet(monitor, status, args.allow_recorded_background)
                    part = folder / f'{repeat}_{method}'
                    rows.extend(run_condition(renderer, monitor, part, method, repeat, target,
                        .5 if args.smoke else STAGE_SECONDS, status, args.allow_recorded_background))
            write_csv(folder / 'TIMINGS.csv', rows)
            assert renderer.quality(folder) == expected
        assert all(sha(name) == value for name, value in protected.items())
        save(folder / 'PROTECTED_AFTER.json', {name: sha(name) for name in protected})
        save(folder / 'INTEGRITY.json', dict(passed=True, protected_files=len(protected),
            models_and_production_unchanged=True, repeated_images_unchanged=True, training_steps=0, saved_render_images=0))
        status.update(state='complete', rows=len(rows), completed_unix=time.time())
        save(folder / 'COMPLETION.json', status)
    except Exception as error:
        status.update(state='failed', error=repr(error))
        raise
    finally:
        monitor.close()
        save(ROOT / 'STATUS.json', status)
    print('GPU_LOAD_RUN_COMPLETE', args.label, len(rows), flush=True)


if __name__ == '__main__':
    main()
