"""局部消融顺序队列；结果增量汇总，异常现场不覆盖。"""
from pathlib import Path
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from common import ROOT, records, load, write


def identity(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid, start_ticks=fields[19], boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def main():
    lock = (ROOT/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(ROOT/'CONTROLLER.json', identity(os.getpid()))
    groups = list(dict.fromkeys(row['group'] for row in records()))
    assert load(ROOT/'SMOKE_VERIFIED.json')['passed']
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    started = time.time()
    finished = 0
    for group in groups:
        base = ROOT/'runs'/group
        completion = base/'GROUP_COMPLETE.json'
        if not completion.exists():
            for attempt in range(1,4):
                folder = base/f'attempt_{attempt:02d}'
                if (folder/'COMPLETE.json').exists():
                    write(completion, dict(group=group, output=str(folder), attempt=attempt))
                    break
                if (folder/'EXIT.json').exists():
                    previous = load(folder/'EXIT.json')
                    assert previous['oom'], f'保留非OOM失败，不盲目重跑：{folder}'
                    continue
                assert not (folder/'PROCESS.json').exists(), f'先核查旧worker身份：{folder}'
                if shutil.disk_usage(ROOT).free < 5*1024**3:
                    raise RuntimeError('DISK_BELOW_5GIB')
                folder.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, '-B', str(ROOT/'worker.py'), '--group', group, '--attempt', str(attempt)]
                with (folder/'worker.log').open('a') as log:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                    write(folder/'PROCESS.json', dict(**identity(child.pid), command=command, started_unix=time.time()))
                    write(ROOT/'STATUS.json', dict(state='running', group=group, attempt=attempt, finished=finished,
                        groups=32, active=identity(child.pid), started_unix=started, updated_unix=time.time()))
                    code = child.wait()
                text = (folder/'worker.log').read_text()[-20000:]
                oom = bool(code and any(x in text.lower() for x in ['cuda out of memory','outofmemoryerror','cublas_status_alloc_failed']))
                write(folder/'EXIT.json', dict(code=code, oom=oom, ended_unix=time.time()))
                if code:
                    if oom and attempt < 3:
                        continue
                    write(ROOT/'STATUS.json', dict(state='failed', group=group, attempt=attempt,
                        finished=finished, groups=32, log=str(folder/'worker.log'), updated_unix=time.time()))
                    raise RuntimeError(f'ABLATION_WORKER_FAILED {group} {attempt} {code}')
                assert load(folder/'COMPLETE.json')['complete']
                write(completion, dict(group=group, output=str(folder), attempt=attempt))
                break
        assert completion.exists(), f'OOM重试已用完：{group}'
        finished += 1
        print('FINISHED_GROUP', finished, 32, group, flush=True)
        subprocess.run([sys.executable, '-B', str(ROOT/'report.py')], check=True, env=env)
    subprocess.run([sys.executable, '-B', str(ROOT/'report.py'), '--final'], check=True, env=env)
    write(ROOT/'STATUS.json', dict(state='complete', finished=32, groups=32, updated_unix=time.time()))


if __name__ == '__main__':
    main()
