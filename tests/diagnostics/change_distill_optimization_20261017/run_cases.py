"""串行运行固定契约案例，记录外部 GPU 进程污染。"""
from pathlib import Path
import json
import subprocess
import sys
import threading
import time

D = Path(__file__).resolve().parent
EVENT = '/home/prmi/zwj/mycartgs/results/change_distill_ablation/20261017-2038/events_supplement/local_loop/event_6207_1'

def processes():
    return subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                    '--format=csv,noheader'], text=True).strip()

def run(stage, branch, folder=None, mode=0, repeats=10):
    out = D / stage / 'raw' / (folder or branch)
    assert not (out / 'runs.json').exists(), str(out)
    assert not processes(), 'GPU is in use: ' + processes()
    out.mkdir(parents=True, exist_ok=True)
    command = [str(D / stage / ('benchmark_native' if mode == 0 else 'benchmark_instrumented')),
               EVENT, str(out), branch, '5', '8192', '4', str(repeats), str(mode), '2']
    (out / 'command.json').write_text(json.dumps(command, indent=2))
    records = []
    with (out / 'stdout.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            records.append({'time': time.time(), 'own_pid': process.pid, 'gpu_processes': processes()})
            time.sleep(0.25)
    (out / 'monitor.json').write_text(json.dumps(records, indent=2))
    contamination = [r for r in records if any(line.split(',')[0].strip() != str(process.pid)
                     for line in r['gpu_processes'].splitlines())]
    assert process.returncode == 0, str(out)
    assert not contamination, str(contamination)
    print('PASS', stage, branch, folder or branch, flush=True)

if __name__ == '__main__':
    for item in sys.argv[1:]:
        fields = item.split(':')
        run(*fields[:3], **({'mode': int(fields[3]), 'repeats': int(fields[4])} if len(fields) > 3 else {}))
