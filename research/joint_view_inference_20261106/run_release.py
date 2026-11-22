"""最终源码的编译后验收与性能复测，复用已完成全轨迹证据。"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import fcntl

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent


def main():
    lock=(ROOT/'reports/release.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(state='running',pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),completed=[])
    stages=[('units','run_units.py',[]),('layers','validate_layers.py',[]),('state','validate_state.py',[]),
        ('pipeline','validate_pipeline.py',[]),('display','validate_display.py',[]),
        ('bench1','benchmark.py',['--round','1']),('bench2','benchmark.py',['--round','2'])]
    p=ROOT/'reports/RELEASE_STATUS.json'
    for name,script,args in stages:
        state.update(stage=name,started=time.time());p.write_text(json.dumps(state,indent=2))
        with (ROOT/'logs'/('release_'+name+'.log')).open('w') as f:
            result=subprocess.run([sys.executable,'-u',str(HERE/script)]+args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:
            state.update(state='failed',returncode=result.returncode);p.write_text(json.dumps(state,indent=2));raise RuntimeError(name)
        state['completed'].append(name)
    state.update(state='complete',stage=None);p.write_text(json.dumps(state,indent=2))


if __name__=='__main__':main()
