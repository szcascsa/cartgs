"""v13最高优先级准备，人工负载仅在渲染阶段运行。"""
import argparse
import csv
import ctypes
import fcntl
import os
from pathlib import Path
import time

from common import *
from control import BudgetController, DUTIES, STAGE_SECONDS
from activity import ActivityMonitor
from phase_gate import RenderLoad, validate_phase_intervals
from reference import Renderer as ReferenceRenderer, METHODS, queue_paused


class Renderer(ReferenceRenderer):
    def __init__(self):
        super().__init__()
        library = ctypes.CDLL('/usr/local/cuda-11.8/lib64/libcudart.so')
        least, greatest = ctypes.c_int(), ctypes.c_int()
        assert library.cudaDeviceGetStreamPriorityRange(ctypes.byref(least), ctypes.byref(greatest)) == 0
        library.cudaStreamCreateWithPriority.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
        self.prepare_handle = ctypes.c_void_p()
        assert library.cudaStreamCreateWithPriority(ctypes.byref(self.prepare_handle), 1, greatest.value) == 0
        self.prepare_stream = torch.cuda.ExternalStream(self.prepare_handle.value)
        self.cuda_library = library
        self.render_stream = torch.cuda.default_stream()
        actual = ctypes.c_int()
        library.cudaStreamGetPriority.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        assert library.cudaStreamGetPriority(self.prepare_stream.cuda_stream, ctypes.byref(actual)) == 0
        assert actual.value == greatest.value
        self.priorities = dict(greatest=greatest.value, least=least.value,
            preparation=actual.value, rendering=self.render_stream.priority, load=least.value)
        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        torch.cuda.synchronize()

    def close(self):
        self.prepare_stream.synchronize()
        self.cuda_library.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        assert self.cuda_library.cudaStreamDestroy(self.prepare_handle) == 0

    def call(self, method, budget=4000, load=None, stage=-1, duty=0.):
        start = time.monotonic()
        stats = {}
        if method == 'cartgs_full':
            k = 35301
            prepared = start
        else:
            with torch.cuda.stream(self.prepare_stream):
                self.events[0].record()
                packet, ids, stats = self.runtime.prepare(self.camera, budget=budget)
                self.events[1].record()
            self.prepare_stream.synchronize()
            prepared = time.monotonic()
            k = len(ids)
            assert k == min(budget, stats['eligible_candidates'])
            assert stats['device_counts'] and stats['shrinking_radix'] and stats['selector_tiled']
            assert stats['compensation_tile'] == (16 if budget >= 1024 else 0)
        phase_start = time.monotonic()
        generation = load.resume(stage, duty) if load else 0
        render_start = time.monotonic()
        try:
            with torch.cuda.stream(self.render_stream):
                self.events[2].record()
                if method == 'cartgs_full':
                    output = self.cart_renderer.render(self.cart_camera, self.cart_pc, self.bg)
                else:
                    output = render_prepared(self.camera, packet, self.bg)
                self.events[3].record()
            self.render_stream.synchronize()
            render_end = time.monotonic()
        finally:
            if load:
                load.pause()
        phase_end = time.monotonic()
        prepare_ms = 1000 * (prepared-start)
        render_ms = 1000 * (render_end-render_start)
        return dict(K=k, candidate_N=stats.get('candidate_gaussians', 35301),
            eligible_N=stats.get('eligible_candidates', 35301), network_ratio=stats.get('ratio'),
            requested_K=35301 if method == 'cartgs_full' else budget,
            prepare_ms=prepare_ms, render_ms=render_ms, total_ms=prepare_ms+render_ms,
            gate_overhead_ms=1000*((render_start-phase_start)+(phase_end-render_end)),
            wall_ms=1000*(phase_end-start), prepare_gpu_ms=0. if method == 'cartgs_full' else self.events[0].elapsed_time(self.events[1]),
            render_gpu_ms=self.events[2].elapsed_time(self.events[3]), load_generation=generation,
            prepare_start_s=start, prepare_end_s=prepared, render_phase_start_s=phase_start,
            render_start_s=render_start, render_end_s=render_end, render_phase_end_s=phase_end), output['render']


def verify_stream_equivalence(renderer, folder):
    rows = []
    with torch.inference_mode():
        for budget in (0, 1, 100, 1000, 1023, 1024, 2000, 3000, 4000, 6632, 10000):
            with torch.cuda.stream(renderer.render_stream):
                expected, expected_ids, _ = renderer.runtime.prepare(renderer.camera, budget=budget)
            renderer.render_stream.synchronize()
            with torch.cuda.stream(renderer.prepare_stream):
                actual, actual_ids, _ = renderer.runtime.prepare(renderer.camera, budget=budget)
            renderer.prepare_stream.synchronize()
            assert torch.equal(expected_ids, actual_ids)
            comparisons = 1
            for key, value in expected['inputs'].items():
                if value is not None:
                    assert torch.equal(value.contiguous().view(torch.uint8), actual['inputs'][key].contiguous().view(torch.uint8)), (budget, key)
                    comparisons += 1
            first = render_prepared(renderer.camera, expected, renderer.bg)['render']
            second = render_prepared(renderer.camera, actual, renderer.bg)['render']
            renderer.render_stream.synchronize()
            assert torch.equal(first.view(torch.uint8), second.view(torch.uint8))
            rows.append(dict(budget=budget, actual_K=len(actual_ids), bitwise_comparisons=comparisons+1, passed=True))
    save(folder/'STREAM_EQUIVALENCE.json', dict(passed=True, cases=rows, priorities=renderer.priorities))


def run_condition(renderer, monitor, folder, method, repeat, target, seconds_per_stage, status):
    folder.mkdir()
    load = RenderLoad(ROOT/'build/librender_load.so', folder/'LOAD_PACKETS.csv')
    rows, boundaries = [], []
    controller = BudgetController(target)
    try:
        for _ in range(20):
            renderer.call(method)
        start = time.monotonic()
        next_request = start
        active_stage = -1
        while time.monotonic()-start < len(DUTIES)*seconds_per_stage:
            now = time.monotonic()
            stage = min(len(DUTIES)-1, int((now-start)//seconds_per_stage))
            if stage != active_stage:
                assert queue_paused(), 'Experiment queue resumed'
                active_stage = stage
                boundaries.append(dict(stage=stage, monotonic_s=now, elapsed_s=now-start))
                status.update(state='timing', method=method, repeat=repeat, stage=stage, budget=controller.budget)
                save(ROOT/'STATUS.json', status)
            budget = controller.budget if method == 'ours_adaptive' else 4000
            record, _ = renderer.call(method, budget, load, stage, DUTIES[stage])
            if method == 'ours_adaptive':
                controller.update(record['total_ms'])
            record.update(method=method, repeat=repeat, stage=stage, target_duty=DUTIES[stage],
                budget=35301 if method == 'cartgs_full' else budget, elapsed_s=now-start,
                completed_s=time.monotonic()-start, monotonic_s=now,
                next_budget=controller.budget, tau_ms=target, over_target=record['total_ms']>target,
                crossed_stage=time.monotonic()-start >= (stage+1)*seconds_per_stage)
            rows.append(record)
            next_request += .005
            delay = next_request-time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_request = time.monotonic()
        end = time.monotonic()
    finally:
        load.close()
        if rows:
            write_csv(folder/'TIMINGS.csv', rows)
    with (folder/'LOAD_PACKETS.csv').open() as stream:
        packets = list(csv.DictReader(stream))
    audit = validate_phase_intervals(rows, packets)
    for stage, duty in enumerate(DUTIES):
        stage_packets = [v for v in packets if int(v['stage']) == stage]
        assert bool(stage_packets) == bool(duty), (stage, duty)
    time.sleep(4)
    telemetry, others = monitor.unexpected(start-2, end+3, {os.getpid()})
    assert len(telemetry) >= int(end-start)+3, 'Missing GPU activity coverage'
    audit.update(start_monotonic_s=start, end_monotonic_s=end, stages=boundaries,
        background_activity=others, pmon_samples=len(telemetry), shared_gpu=True)
    save(folder/'ACCEPTANCE.json', audit)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    lock = (ROOT/'reports/run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    folder = ROOT/'runs'/args.label
    folder.mkdir()
    status = dict(state='starting', process=identity(), label=args.label, smoke=args.smoke)
    save(ROOT/'STATUS.json', status)
    monitor = ActivityMonitor(folder)
    renderer = None
    try:
        protected = json.loads((ROOT/'reports/PROTECTED_BEFORE.json').read_text())
        assert all(sha(name)==value for name,value in protected.items())
        assert sha(V13_BINARY)=='eb0195413f9b44a79fe93f4f1e313437469a6c367adf289b69006e190339e15e'
        while len(monitor.samples)<3:
            if monitor.error:
                raise RuntimeError(monitor.error)
            time.sleep(.2)
        assert queue_paused(), 'Queue must stay held'
        cpu_priority = dict(before=os.getpriority(os.PRIO_PROCESS, 0), requested=-20)
        try:
            os.setpriority(os.PRIO_PROCESS, 0, -20)
            cpu_priority['set_success']=True
        except PermissionError as error:
            cpu_priority.update(set_success=False, error=str(error))
        cpu_priority['actual']=os.getpriority(os.PRIO_PROCESS, 0)
        with torch.inference_mode(False), torch.no_grad():
            renderer = Renderer()
        with torch.inference_mode():
            verify_stream_equivalence(renderer, folder)
            expected = renderer.quality(folder)
            old = ROOT/'reports/V11_QUALITY.csv'
            with old.open() as stream:
                previous = {(v['method'],int(v['budget'])):v['image_sha256'] for v in csv.DictReader(stream)}
            assert expected == previous, 'v13 image differs from accepted v11 fixed-view images'
            for _ in range(100):
                renderer.call('ours_fixed4000')
                renderer.call('cartgs_full')
            calibration = [renderer.call('ours_fixed4000')[0]['total_ms'] for _ in range(200)]
            target = 1.1*float(np.percentile(calibration,95))
            save(folder/'CALIBRATION.json', dict(samples_ms=calibration, p95_ms=target/1.1, tau_ms=target))
            save(folder/'ENVIRONMENT.json', dict(binary_sha256=sha(V13_BINARY), version='v13', frame=396,
                own_checkpoint=str(CHECKPOINT), cartgs_checkpoint=str(VANILLA), priorities=renderer.priorities,
                cpu_priority=cpu_priority, target_ms=target, duties=DUTIES,
                stage_seconds=.5 if args.smoke else STAGE_SECONDS, request_period_ms=5,
                methods=METHODS, artificial_load='render_only_in_process_low_priority_stream',
                gate_overhead_excluded_from_total=True, shared_gpu=True, gpu=gpu_state('before'),
                cuda=torch.version.cuda, torch=torch.__version__, tf32=False))
            rows=[]
            for repeat in range(1 if args.smoke else 3):
                for method in METHODS[repeat:]+METHODS[:repeat]:
                    rows.extend(run_condition(renderer, monitor, folder/f'{repeat}_{method}', method, repeat,
                        target, .5 if args.smoke else STAGE_SECONDS, status))
            write_csv(folder/'TIMINGS.csv', rows)
            assert renderer.quality(folder)==expected
        assert all(sha(name)==value for name,value in protected.items())
        save(folder/'INTEGRITY.json', dict(passed=True, protected_files=len(protected),
            image_bitwise_before_after=True, v13_vs_v11_images_bitwise=True,
            artificial_load_prepare_overlap=0, saved_render_images=0, training_steps=0))
        status.update(state='complete', rows=len(rows), completed_unix=time.time())
        save(folder/'COMPLETION.json', status)
    except Exception as error:
        status.update(state='failed', error=repr(error))
        raise
    finally:
        if renderer is not None:
            renderer.close()
        monitor.close()
        save(ROOT/'STATUS.json', status)
    print('RENDER_ONLY_V13_COMPLETE', args.label, len(rows), flush=True)


if __name__=='__main__':
    main()
