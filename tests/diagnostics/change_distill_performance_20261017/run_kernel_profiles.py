"""只采集独立 CD 的一个 epoch；不启动 SLAM，不安装 profiler。"""
from pathlib import Path
import json
import subprocess
import csv
import re
from analyze import parse_trace
from run_matrix import D, EVENT, RAW, command, wait_idle

nsys = Path('/usr/local/cuda-11.8/bin/nsys')
ncu = Path('/usr/local/cuda-11.8/bin/ncu')
results = {}
for branch in ['selector', 'tf']:
    wait_idle()
    output = RAW / ('nsys_' + branch)
    runner = [D / 'benchmark_instrumented', EVENT, RAW / ('nsys_case_' + branch),
              branch, 1, 8192, 4, 0, 3, 2]
    args = [nsys, 'profile', '--trace=cuda,nvtx,cublas,osrt', '--sample=none',
            '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
            '--force-overwrite=true', '--output=' + str(output), *runner]
    try:
        command(args, RAW / ('nsys_' + branch + '.log'))
        results[branch] = {'nsys': 'completed'}
        command([nsys, 'stats', '--report', 'gpukernsum,cudaapisum,nvtxsum',
                 '--format', 'csv', '--force-export=true', str(output) + '.nsys-rep'],
                RAW / ('nsys_' + branch + '_stats.csv'))
    except Exception as error:
        results.setdefault(branch, {})['nsys_error'] = str(error)

# Select one measured backward hotspot per branch, rather than guessing a kernel.
# NCU permission failures are recorded, never worked around by changing the driver.
for branch in ['tf', 'selector']:
    trace = RAW / 'cases' / f'{branch}_e1_b8192_r4_m2/profile/trace.json'
    parse_trace(trace, branch)
    with (RAW / (branch + '_kernels.csv')).open() as file:
        hotspots = [row for row in csv.DictReader(file) if row['backward'] == 'True']
    if not hotspots:
        results.setdefault(branch, {})['ncu_error'] = 'No measured backward CUDA kernel; skipped'
        continue
    kernel = hotspots[0]['kernel']
    results.setdefault(branch, {})['ncu_selected_hotspot'] = kernel
    wait_idle()
    runner = [D / 'benchmark_instrumented', EVENT, RAW / ('ncu_case_' + branch),
              branch, 1, 8192, 4, 0, 3, 2]
    args = [ncu, '--profile-from-start', 'off', '--kernel-name-base', 'demangled',
            '--kernel-name', 'regex:' + re.escape(kernel), '--launch-count', '2',
            '--metrics', 'gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed',
            '--csv', '--log-file',
            RAW / ('ncu_' + branch + '.csv'), *runner]
    try:
        command(args, RAW / ('ncu_' + branch + '.log'))
        content=(RAW / ('ncu_' + branch + '.csv')).read_text()
        results.setdefault(branch, {})['ncu'] = ('permission_denied_ERR_NVGPUCTRPERM' if 'ERR_NVGPUCTRPERM' in content else
                                               'no_kernel_captured' if 'No kernels were profiled' in content else 'completed')
    except Exception as error:
        results.setdefault(branch, {})['ncu_error'] = str(error)
(RAW / 'kernel_profile_status.json').write_text(json.dumps(results, indent=2))
