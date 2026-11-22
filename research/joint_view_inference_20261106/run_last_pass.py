"""最终调度修正后串行验收；失败不继续测量或篡改已有结果。"""
from pathlib import Path
import subprocess
import json
import os
import sys
import time
import fcntl
import shutil

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent


def main():
    lock=(ROOT/'reports/last_pass.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(state='running',pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),completed=[])
    p=ROOT/'reports/LAST_PASS_STATUS.json'
    for name in ['run_release.py','run_final_checks.py','benchmark_moving.py','validate_display.py']:
        if name=='validate_display.py':shutil.copyfile(ROOT/'reports/DISPLAY_VALIDATION.json',ROOT/'reports/DISPLAY_REPEAT1.json')
        state.update(stage=name,started=time.time());p.write_text(json.dumps(state,indent=2))
        with (ROOT/'logs'/('last_'+name+'.log')).open('w') as f:
            child=subprocess.run([sys.executable,'-u',str(HERE/name)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if child.returncode:
            state.update(state='failed',returncode=child.returncode);p.write_text(json.dumps(state,indent=2));raise RuntimeError(name)
        state['completed'].append(name)
    state.update(state='complete',stage=None);p.write_text(json.dumps(state,indent=2))


if __name__=='__main__':main()
