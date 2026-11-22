"""v27完整轨迹、四档实测占空比与三个渲染入口的配对测试。"""
import argparse
import csv
import ctypes
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
os.environ['CUDA_HOME'] = '/usr/local/cuda-11.8'
os.environ['TORCH_EXTENSIONS_DIR'] = str(ROOT / 'build')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path[:0] = [str(ROOT / 'source/python'), str(ROOT / 'source/evaluation'),
               str(ROOT / 'source/evaluation/diff-gaussian-rasterization')]
import numpy as np
import torch
from budget_viewer import BudgetScene, load_views
from gaussian_model import GaussianModel
from renderer import render_prepared
from utils import MiniCam, focal2fov
from phase_gate import RenderLoad, validate_phase_intervals

CHECKPOINT = Path('/home/prmi/zwj/experiments/view_conditioned_li_20261106/runs/full_both_li/23560_shutdown/ply')
VANILLA = Path('/home/prmi/zwj/realLOD/saved_experiments/vanilla_all_datasets_20261031_2011/jobs/tum/rgbd/fr1_xyz/repeat1/seed0/attempt_2/model/10400_shutdown/ply')
DATASET = Path('/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz')
V27 = ROOT / 'build/mycartgs_deployment_fp32_v27/mycartgs_deployment_fp32_v27.so'
V27_SHA = '8f69753da6b7b346296e54b258a1d74fd81403c49e436093a1217ba7c9a14d89'
METHODS = ('cartgs_full', 'ours_full', 'ours_adaptive')
DUTIES = (0., .4, .6, .8)
BUDGETS = (4000, 2000, 1000, 500)
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name, path, package=False):
    spec = importlib.util.spec_from_file_location(name, path,
        submodule_search_locations=[str(path.parent)] if package else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def gpu_state():
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw',
                                   '--format=csv,noheader'], text=True).strip()
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                         '--format=csv,noheader,nounits'], text=True)
    rows = [dict(pid=int(row[0]), name=row[1].strip(), memory=row[2].strip())
            for row in csv.reader(processes.splitlines()) if row]
    return dict(unix=time.time(), monotonic_s=time.monotonic(), gpu=gpu, processes=rows,
                foreign=[row for row in rows if row['pid'] != os.getpid()])


class Monitor:
    def __init__(self, folder):
        self.stop = threading.Event()
        self.error = None
        self.samples = []
        self.folder = folder
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        try:
            with (self.folder / 'GPU_MONITOR.jsonl').open('w', buffering=1) as stream:
                while not self.stop.is_set():
                    record = gpu_state()
                    self.samples.append(record)
                    stream.write(json.dumps(record) + '\n')
                    if record['foreign']:
                        raise RuntimeError('Foreign CUDA process appeared ' + str(record['foreign']))
                    self.stop.wait(1.)
        except Exception as error:
            self.error = repr(error)

    def check(self):
        if self.error:
            raise RuntimeError(self.error)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)
        assert not self.thread.is_alive()
        self.check()


class Renderer:
    def __init__(self):
        assert sha(V27) == V27_SHA
        native = load_module('mycartgs_deployment_fp32_v27', V27)
        frozen = load_module('mycartgs_frozen_inference_v3',
            ROOT / 'build/mycartgs_frozen_inference_v3/mycartgs_frozen_inference_v3.so')
        import deployment_runtime
        import frozen_inference_cuda
        deployment_runtime.deployment_extension = lambda: native
        frozen_inference_cuda.extension = lambda: frozen
        self.scene = BudgetScene(CHECKPOINT, DATASET, 4000, 0, None, 'pytorch')
        self.runtime = deployment_runtime.DeploymentRenderer(self.scene.model, self.scene.gaussians, self.scene.config)
        calibration, cart_views = load_views(VANILLA, DATASET)
        assert calibration == self.scene.calibration
        by_timestamp = {view['timestamp']: view for view in cart_views}
        assert len(by_timestamp) == len(cart_views)
        assert len(self.scene.views) == 792
        assert all(view['timestamp'] in by_timestamp for view in self.scene.views)
        self.cart_pc = GaussianModel(3)
        ply = list((VANILLA / 'point_cloud').glob('iteration_*/point_cloud.ply'))
        assert len(ply) == 1
        self.cart_pc.load_ply(str(ply[0]))
        raster = load_module('original_raster', ROOT / 'source/original_raster/__init__.py', True)
        self.cart_renderer = load_module('original_renderer', ROOT / 'source/original_renderer.py')
        self.cart_renderer.GaussianRasterizationSettings = raster.GaussianRasterizationSettings
        self.cart_renderer.GaussianRasterizer = raster.GaussianRasterizer
        self.cameras = []
        self.cart_cameras = []
        for view in self.scene.views:
            self.cameras.append(self.camera(calibration, view))
            self.cart_cameras.append(self.camera(calibration, by_timestamp[view['timestamp']]))
        self.library = ctypes.CDLL('/usr/local/cuda-11.8/lib64/libcudart.so')
        least, greatest = ctypes.c_int(), ctypes.c_int()
        assert self.library.cudaDeviceGetStreamPriorityRange(ctypes.byref(least), ctypes.byref(greatest)) == 0
        self.handle = ctypes.c_void_p()
        self.library.cudaStreamCreateWithPriority.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
        assert self.library.cudaStreamCreateWithPriority(ctypes.byref(self.handle), 1, greatest.value) == 0
        self.prepare_stream = torch.cuda.ExternalStream(self.handle.value)
        self.library.cudaStreamGetPriority.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        actual = ctypes.c_int()
        assert self.library.cudaStreamGetPriority(self.handle, ctypes.byref(actual)) == 0
        assert actual.value == greatest.value
        self.render_stream = torch.cuda.default_stream()
        self.priorities = dict(preparation=actual.value, highest=greatest.value,
                               rendering=self.render_stream.priority)
        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        torch.cuda.synchronize()
        assert self.runtime.total == 13298 and len(self.cart_pc.get_xyz) == 35301

    @staticmethod
    def camera(calibration, view):
        return MiniCam(calibration['width'], calibration['height'],
            focal2fov(calibration['fx'], calibration['width']),
            focal2fov(calibration['fy'], calibration['height']), np.linalg.inv(np.asarray(view['pose'])))

    def prepare(self, frame, method, budget):
        if method == 'ours_full':
            packet, ids = self.runtime.full()
            return packet, ids, dict(candidate_gaussians=self.runtime.total,
                                     eligible_candidates=self.runtime.total, ratio=None)
        packet, ids, stats = self.runtime.prepare(self.cameras[frame], budget=budget)
        assert len(ids) == min(budget, stats['eligible_candidates'])
        assert stats['device_counts'] and stats['shrinking_radix'] and stats['graph_replay']
        return packet, ids, stats

    def call(self, frame, method, budget, duty=0., load=None):
        start = time.monotonic()
        if method == 'cartgs_full':
            prepared = start
            count = len(self.cart_pc.get_xyz)
            stats = dict(candidate_gaussians=count, eligible_candidates=count, ratio=None)
        else:
            with torch.cuda.stream(self.prepare_stream):
                self.events[0].record()
                packet, ids, stats = self.prepare(frame, method, budget)
                self.events[1].record()
            self.prepare_stream.synchronize()
            prepared = time.monotonic()
            count = len(ids)
        phase_start = time.monotonic()
        generation = load.resume(int(frame * 4 // len(self.cameras)), duty) if load else 0
        render_start = time.monotonic()
        try:
            with torch.cuda.stream(self.render_stream):
                self.events[2].record()
                if method == 'cartgs_full':
                    output = self.cart_renderer.render(self.cart_cameras[frame], self.cart_pc, self.scene.background)
                else:
                    output = render_prepared(self.cameras[frame], packet, self.scene.background, return_radii=True)
                self.events[3].record()
            self.render_stream.synchronize()
            render_end = time.monotonic()
        finally:
            if load:
                load.pause()
        phase_end = time.monotonic()
        prepare_ms = 1000 * (prepared - start)
        render_ms = 1000 * (render_end - render_start)
        record = dict(frame=frame+1, timestamp=self.scene.views[frame]['timestamp'], method=method,
            stage=frame*4//len(self.cameras), target_duty=duty, requested_K=count if method != 'ours_adaptive' else budget,
            K=count, candidate_N=stats['candidate_gaussians'], eligible_N=stats['eligible_candidates'],
            network_ratio=stats.get('ratio'), graph_replay=stats.get('graph_replay',False),
            prepare_ms=prepare_ms, render_ms=render_ms, pipeline_ms=prepare_ms+render_ms,
            gate_overhead_ms=1000*((render_start-phase_start)+(phase_end-render_end)),
            request_ms=1000*(phase_end-start),
            prepare_gpu_ms=0. if method=='cartgs_full' else self.events[0].elapsed_time(self.events[1]),
            render_gpu_ms=self.events[2].elapsed_time(self.events[3]), load_generation=generation,
            prepare_start_s=start, prepare_end_s=prepared, render_phase_start_s=phase_start,
            render_start_s=render_start, render_end_s=render_end, render_phase_end_s=phase_end)
        record['radii_positive'] = int((output['radii'] > 0).sum())
        assert bool(torch.isfinite(output['render']).all())
        return record, output['render']

    def close(self):
        self.prepare_stream.synchronize()
        self.library.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        assert self.library.cudaStreamDestroy(self.handle) == 0


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def audit_load(folder, frames, hardware):
    packets = read_csv(folder / 'LOAD_PACKETS.csv')
    phases = read_csv(folder / 'LOAD_PACKETS.csv.phases.csv')
    assert len(phases) == len(frames)
    audit = validate_phase_intervals(frames, packets)
    for packet in packets:
        assert int(packet['unique_sms']) == hardware['sm_count']
        assert int(packet['blocks_each_sm']) == hardware['resident_blocks_per_sm']
    for phase in phases:
        duty = float(phase['target_duty'])
        assert float(phase['measured_resident_duty']) <= duty + 1e-8
        assert float(phase['charged_duty']) <= duty + 1e-8
        assert (int(phase['pulses']) > 0) == (duty > 0)
    active = [phase for phase in phases if float(phase['target_duty']) > 0]
    error = max(abs(float(phase['measured_resident_duty'])-float(phase['target_duty'])) for phase in active)
    audit.update(all_sm_and_thread_slots=True, actual_duty_cap=True, maximum_target_error=error,
                 target_match_within_five_percentage_points=error <= .05)
    save(folder / 'LOAD_ACCEPTANCE.json', audit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    lock = (ROOT / 'reports/run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    folder = ROOT / 'runs' / args.label
    folder.mkdir()
    before = gpu_state()
    assert not before['foreign'], before
    protected = json.loads((ROOT / 'reports/PROTECTED_BEFORE.json').read_text())
    assert all(sha(Path(name)) == value for name, value in protected.items())
    state = dict(state='starting', label=args.label, pid=os.getpid(),
        start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    save(ROOT / 'reports/STATUS.json', state)
    monitor = Monitor(folder)
    renderer = None
    try:
        with torch.inference_mode(False), torch.no_grad():
            renderer = Renderer()
        boundaries = [0,197,198,395,396,593,594,791]
        frames = boundaries if args.smoke else list(range(len(renderer.cameras)))
        signatures = {}
        with torch.inference_mode():
            for frame in boundaries:
                for method in METHODS:
                    for _ in range(3):
                        renderer.call(frame, method, BUDGETS[frame*4//792])
            warmed = time.monotonic()
            while time.monotonic()-warmed < 2:
                renderer.call(396,'cartgs_full',1000)
                renderer.call(396,'ours_adaptive',1000)
            baseline = []
            for frame in frames:
                monitor.check()
                for method in METHODS:
                    record, image = renderer.call(frame, method, BUDGETS[frame*4//792])
                    record['repeat'] = -1
                    baseline.append(record)
                    if frame in boundaries:
                        signatures[frame,method] = hashlib.sha256(image.cpu().numpy().tobytes()).hexdigest()
            with (folder / 'NO_LOAD.csv').open('w',newline='') as stream:
                writer = csv.DictWriter(stream,fieldnames=list(baseline[0]))
                writer.writeheader()
                writer.writerows(baseline)
            expected = {f'{frame}/{method}':value for (frame,method),value in signatures.items()}
            save(folder / 'REFERENCE_SIGNATURES.json', expected)
            load = RenderLoad(ROOT / 'build/librender_load.so', folder / 'LOAD_PACKETS.csv')
            hardware = load.information
            assert hardware['maximum_duty'] == .8
            assert hardware['resident_threads_per_sm'] == hardware['maximum_threads_per_sm']
            assert hardware['preparation_highest_priority'] == renderer.priorities['preparation']
            save(folder / 'ENVIRONMENT.json', dict(v27_sha256=sha(V27), torch=torch.__version__,cuda=torch.version.cuda,
                checkpoint=str(CHECKPOINT),cartgs_checkpoint=str(VANILLA),views=len(renderer.cameras),
                duties=DUTIES,budgets=BUDGETS,priorities=renderer.priorities,hardware=hardware,gpu_before=before,
                full_path='uncompensated_master',camera_path='matched_timestamps_each_saved_slam_pose',
                repeats=1 if args.smoke else 3,smoke=args.smoke))
            records = []
            try:
                with (folder / 'TIMINGS.csv').open('w',newline='',buffering=1) as stream:
                    writer = None
                    for repeat in range(1 if args.smoke else 3):
                        for frame in frames:
                            monitor.check()
                            stage = frame*4//792
                            rotation = (repeat+frame)%3
                            for method in METHODS[rotation:]+METHODS[:rotation]:
                                record,image = renderer.call(frame,method,BUDGETS[stage],DUTIES[stage],load)
                                record['repeat'] = repeat
                                if frame in boundaries:
                                    assert hashlib.sha256(image.cpu().numpy().tobytes()).hexdigest() == signatures[frame,method]
                                if writer is None:
                                    writer = csv.DictWriter(stream,fieldnames=list(record))
                                    writer.writeheader()
                                writer.writerow(record)
                                records.append(record)
                            if frame%66 == 0 or frame in boundaries:
                                state.update(state='timing',repeat=repeat,frame=frame+1,stage=stage,rows=len(records))
                                save(ROOT/'reports/STATUS.json',state)
                                print('PROGRESS',repeat,frame+1,stage,len(records),flush=True)
            finally:
                load.close()
            audit_load(folder, records, hardware)
        monitor.close()
        assert all(sha(Path(name)) == value for name,value in protected.items())
        save(folder / 'INTEGRITY.json', dict(passed=True, protected_files=len(protected),
            all_spot_pixels_bitwise=True,spot_views=boundaries,prepare_overlap=0,saved_render_images=0))
        assert len(records) == len(frames)*3*(1 if args.smoke else 3)
        state.update(state='complete',rows=len(records),views=len(frames))
        print('TRAJECTORY_COMPLETE',args.label,len(records),flush=True)
    except Exception as error:
        state.update(state='failed',error=repr(error))
        raise
    finally:
        if renderer:
            renderer.close()
        if monitor.thread.is_alive():
            monitor.close()
        save(ROOT/'reports/STATUS.json',state)


if __name__ == '__main__':
    main()
