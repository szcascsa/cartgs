"""Fail-closed CUDA gates A/B/C/D. No online sequence runs in this driver."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CMAKE = Path('/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake')
CTEST = CMAKE.with_name('ctest')
BUILD = ROOT / 'build-cuda118-sm86-gcc9'


def run(command, log, timeout=1200):
    begin = time.monotonic()
    with (OUT / log).open('w') as stream:
        result = subprocess.run([str(x) for x in command], cwd=ROOT,
                                stdout=stream, stderr=subprocess.STDOUT,
                                timeout=timeout, env=os.environ.copy())
    record = {'command': [str(x) for x in command], 'exit': result.returncode,
              'seconds': time.monotonic()-begin, 'log': str(OUT/log)}
    print(json.dumps(record), flush=True)
    if result.returncode:
        raise RuntimeError(f'Gate failed; see {OUT/log}')
    return record


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'Real CUDA gates cannot be replaced by skipped tests'
    os.environ['PATH'] = '/usr/local/cuda-11.8/bin:' + os.environ['PATH']
    report = {'commit': args.commit, 'torch': torch.__version__,
              'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
              'complete': False, 'commands': []}
    try:
        report['commands'].append(run([CMAKE,'--build',BUILD,'-j2'],'build_gates.log'))
        for stage, pattern in [('A','^ultralight_mode2_(config|full_parity)$'),
                               ('B','^ultralight_mode2_(stages|clocks|special)$')]:
            report['commands'].append(run([CTEST,'--test-dir',BUILD,'-R',pattern,
                                           '--output-on-failure','--timeout','120'],f'gate_{stage}.log'))
            assert 'Skipped' not in (OUT/f'gate_{stage}.log').read_text()
        report['commands'].append(run([CTEST,'--test-dir',BUILD,'--output-on-failure',
                                       '--timeout','120','-j1'],'gate_C.log'))
        report['commands'].append(run([ROOT/'bin/ultralight_mode2_mapping_test','benchmark'],
                                       'gate_D.log',300))
        lines = (OUT/'gate_D.log').read_text().splitlines()
        report['microbenchmark'] = []
        for line in lines:
            if line.startswith('[Mode2 MICRO]'):
                fields = dict(re.findall(r'(\w+)=([^ ]+)',line))
                fields['complete_mean_ms'] = float(fields['complete_mean_ms'])
                fields['host_p95_ms'] = float(fields['host_p95_ms'])
                report['microbenchmark'].append(fields)
        assert len(report['microbenchmark']) == 5, 'Missing benchmark variant'
        report['complete'] = True
    finally:
        report['source_sha256'] = {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
                                  for f in ['src/gaussian_mapper.cpp','include/gaussian_mapper.h',
                                            'include/mapping_profiler.h','tests/ultralight_mode2_mapping_test.cpp',
                                            'CMakeLists.txt']}
        (OUT/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: A/B/C/D; online runs may now be admitted',flush=True)


if __name__ == '__main__':
    main()
