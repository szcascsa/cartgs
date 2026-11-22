"""同一真实事件的独立分支性能矩阵；每个 run 由 native runner 完整恢复。"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time

D = Path(__file__).resolve().parent
EVENT = Path('/home/prmi/zwj/mycartgs/results/change_distill_ablation/20261017-2038/events_supplement/local_loop/event_6207_1')
RAW = D / 'raw'
RAW.mkdir(exist_ok=True)


def command(args, output, timeout=600):
    with output.open('w') as log:
        result = subprocess.run(list(map(str, args)), cwd=D, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    output.with_suffix(output.suffix + '.exit').write_text(str(result.returncode))
    if result.returncode:
        raise RuntimeError(f'{result.returncode}: {output}')


def gpu_processes():
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                      '--format=csv,noheader,nounits'], text=True)
    return output.strip()


def wait_idle():
    # Do not suspend, terminate, or reconfigure unrelated experiments.
    idle = 0
    with (RAW / 'idle_gate.jsonl').open('a') as log:
        while idle < 3:
            processes = gpu_processes()
            log.write(json.dumps({'time': time.time(), 'processes': processes}) + '\n')
            log.flush()
            idle = idle + 1 if not processes else 0
            if idle < 3:
                time.sleep(5)


def environment():
    import torch
    data = {'torch': torch.__version__, 'cuda': torch.version.cuda, 'torch_config': torch.__config__.show(),
            'python': os.sys.version, 'gpu': torch.cuda.get_device_name(),
            'capability': torch.cuda.get_device_capability(), 'allow_tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'allow_tf32_cudnn': torch.backends.cudnn.allow_tf32,
            'nsys': '/usr/local/cuda-11.8/bin/nsys', 'ncu': '/usr/local/cuda-11.8/bin/ncu'}
    data['nvidia_smi'] = subprocess.check_output(['nvidia-smi', '-q'], text=True)
    data['compiler'] = subprocess.check_output(['/usr/bin/g++', '--version'], text=True)
    data['nvcc'] = subprocess.check_output(['/usr/local/cuda-11.8/bin/nvcc', '--version'], text=True)
    (RAW / 'environment.json').write_text(json.dumps(data, indent=2))
    hashes = {}
    for name in ['event.json', 'snapshot.pt', 'selector.pt', 'transform_field.pt']:
        with (EVENT / name).open('rb') as f:
            hashes[name] = hashlib.file_digest(f, 'sha256').hexdigest()
    (RAW / 'input_sha256.json').write_text(json.dumps(hashes, indent=2))
    # Release the inspection process's CUDA context before the idle gate.
    return data


def execute_case(branch, epochs, batch, ratios, mode=0, repeats=10, warmups=2, gate=True):
    name = f'{branch}_e{epochs}_b{batch}_r{ratios}_m{mode}'
    folder = RAW / 'cases' / name
    if (folder / 'runs.json').exists():
        return folder
    if gate:
        wait_idle()
    folder.mkdir(parents=True, exist_ok=True)
    executable = D / ('benchmark_native' if mode == 0 else 'benchmark_instrumented')
    args = [executable, EVENT, folder, branch, epochs, batch, ratios, repeats, mode, warmups]
    print('RUN', name, flush=True)
    samples = []
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            samples.append({'time': time.time(), 'own_pid': process.pid, 'compute_processes': gpu_processes()})
            stop.wait(0.25)

    with (folder / 'stdout.log').open('w') as log:
        process = subprocess.Popen(list(map(str, args)), cwd=D, stdout=log, stderr=subprocess.STDOUT)
        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.start()
        try:
            code = process.wait(timeout=600)
            (folder / 'stdout.log.exit').write_text(str(code))
            if code:
                raise RuntimeError(f'{code}: {folder}')
        finally:
            stop.set()
            monitor_thread.join()
            (folder / 'gpu_process_monitor.json').write_text(json.dumps(samples, indent=2))
    # A new unrelated CUDA process invalidates the case. Preserve rejected data.
    foreign = []
    for sample in samples:
        for line in sample['compute_processes'].splitlines():
            if line and int(line.split(',')[0]) != process.pid:
                foreign.append(line)
    if foreign:
        (folder / 'CONTAMINATED.json').write_text(json.dumps(sorted(set(foreign)), indent=2))
        raise RuntimeError('Foreign GPU process detected; case must be repeated: ' + name)
    return folder


def matrix():
    for branch in ['selector', 'tf']:
        for epochs in [1, 2, 5, 10, 20]:
            execute_case(branch, epochs, 8192, 4)
        for batch in [2048, 4096, 8192, 16384, 32768]:
            execute_case(branch, 5, batch, 4)
        for ratios in [1, 2, 4]:
            execute_case(branch, 5, 8192, ratios)
        for batch in [2048, 4096, 8192, 16384, 32768]:
            execute_case(branch, 5, batch, 4, mode=1, repeats=1)
        for epochs in [1, 20]:
            execute_case(branch, epochs, 8192, 4, mode=1, repeats=1)
        for ratios in [1, 2]:
            execute_case(branch, 5, 8192, ratios, mode=1, repeats=1)
        execute_case(branch, 1, 8192, 4, mode=2, repeats=0)
    (RAW / 'matrix_complete.json').write_text(json.dumps({'complete': True, 'warm_repeats': 10}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.environment:
        environment()
    elif args.smoke:
        for branch in ['selector', 'tf']:
            execute_case(branch, 1, 8192, 4, mode=1, repeats=1, warmups=0, gate=False)
    else:
        matrix()
