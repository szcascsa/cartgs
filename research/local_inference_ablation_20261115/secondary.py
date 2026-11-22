"""并行评价队尾八组TUM；主队列抵达时直接复用完成标记。"""
from pathlib import Path
import fcntl
import os
import subprocess
import sys
import time
from common import ROOT, records, load, write
from controller import identity


def main():
    lock = (ROOT/'secondary.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    groups = list(dict.fromkeys(row['group'] for row in records()))[-8:]
    assert all(group.startswith('TUM_RGBD/') for group in groups)
    write(ROOT/'SECONDARY_CONTROLLER.json', dict(**identity(os.getpid()), groups=groups))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    for group in groups:
        base = ROOT/'runs'/group
        completion = base/'GROUP_COMPLETE.json'
        if completion.exists():
            continue
        # 主队列已认领的任务不再启动，防止重复读取同一输出目录。
        if list(base.glob('attempt_*/PROCESS.json')):
            continue
        folder = base/'attempt_01'
        folder.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-B', str(ROOT/'worker.py'), '--group', group]
        with (folder/'worker.log').open('a') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
            write(folder/'PROCESS.json', dict(**identity(child.pid), command=command, started_unix=time.time(), owner='secondary'))
            write(ROOT/'SECONDARY_STATUS.json', dict(state='running', group=group, active=identity(child.pid), updated_unix=time.time()))
            code = child.wait()
        text = (folder/'worker.log').read_text()[-20000:]
        oom = bool(code and any(token in text.lower() for token in ['cuda out of memory','outofmemoryerror','cublas_status_alloc_failed']))
        write(folder/'EXIT.json', dict(code=code, oom=oom, ended_unix=time.time()))
        if code:
            write(ROOT/'SECONDARY_STATUS.json', dict(state='failed', group=group, log=str(folder/'worker.log'), updated_unix=time.time()))
            raise RuntimeError(f'SECONDARY_EVALUATOR_FAILED {group} {code}')
        assert load(folder/'COMPLETE.json')['complete']
        write(completion, dict(group=group, output=str(folder), attempt=1))
        print('SECONDARY_GROUP_COMPLETE', group, flush=True)
    write(ROOT/'SECONDARY_STATUS.json', dict(state='complete', updated_unix=time.time()))


if __name__ == '__main__':
    main()
