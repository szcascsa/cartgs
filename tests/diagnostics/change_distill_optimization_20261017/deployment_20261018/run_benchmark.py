from pathlib import Path
import json
import os
import subprocess
import sys
import time

D = Path(__file__).resolve().parent
ROOT = Path('/home/prmi/zwj/mycartgs')
EVENT = ROOT / 'results/change_distill_ablation/20261017-2038/events_supplement/local_loop/event_6207_1'
phase = sys.argv[1]
environment = os.environ.copy()
library = D / 'backup/lib' if phase == 'baseline' else ROOT / 'lib'
environment['LD_LIBRARY_PATH'] = ':'.join(str(p) for p in [library,
    ROOT / 'third_party/libtorch-cu118/libtorch/lib', Path('/usr/local/cuda-11.8/lib64'),
    ROOT / 'third_party/ORB-SLAM3/lib'])
def processes():
    return subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                    '--format=csv,noheader'], text=True).strip()
for branch in ['selector', 'tf', 'both']:
    assert not processes(), processes()
    out = D / phase / 'raw' / branch
    assert not out.exists(), str(out)
    out.mkdir(parents=True)
    command = [str(D / 'benchmark_production'), str(EVENT), str(out), branch, '5', '8192', '4', '10', '0', '2']
    (out / 'command.json').write_text(json.dumps(command, indent=2))
    monitor = []
    with (out / 'stdout.log').open('w') as log:
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            monitor.append({'time': time.time(), 'own_pid': process.pid, 'gpu_processes': processes()})
            time.sleep(0.3)
    (out / 'monitor.json').write_text(json.dumps(monitor, indent=2))
    assert process.returncode == 0, str(out)
    assert all(all(s.split(',')[0].strip() == str(process.pid) for s in r['gpu_processes'].splitlines()) for r in monitor)
    print(phase, branch, 'PASS', flush=True)
