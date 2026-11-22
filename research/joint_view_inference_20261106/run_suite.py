"""串行验收和两轮测量，保存进程身份，失败停机保留日志。"""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent


def save(v):
    p=ROOT/'reports/SUITE_STATUS.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2));q.replace(p)


def main():
    lock=(ROOT/'reports/suite.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(state='running',pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),completed=[])
    for name,args in [('validate_state.py',[]),('benchmark.py',['--round','1']),('benchmark.py',['--round','2'])]:
        label=name+('_'+args[-1] if args else '');state.update(stage=label,started=time.time());save(state)
        with (ROOT/'logs'/(label+'.log')).open('w') as f:
            p=subprocess.run([sys.executable,'-u',str(HERE/name)]+args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if p.returncode:state.update(state='failed',returncode=p.returncode);save(state);raise RuntimeError(label)
        state['completed'].append(label)
    state.update(state='complete',stage=None);save(state)


if __name__=='__main__':main()
