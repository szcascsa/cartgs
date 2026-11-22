"""拓扑、Nsight、独立显存测量与全轨迹依次执行，失败保留现场。"""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent


def main():
    lock=(ROOT/'reports/final_checks.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((ROOT/'reports/SUITE_STATUS.json').read_text())['state']=='complete'
    state=dict(state='running',pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),completed=[])
    nsys='/usr/local/cuda-11.8/bin/nsys';trace=ROOT/'reports/INFERENCE_NSYS'
    stages=[('topology',[sys.executable,'-u',str(HERE/'validate_topology.py')]),
        ('nsys_profile',[nsys,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node',
                         '--force-overwrite=true','-o',str(trace),sys.executable,str(HERE/'profile.py')]),
        ('nsys_export',[nsys,'export','--type=sqlite','--force-overwrite=true','-o',str(trace)+'.sqlite',str(trace)+'.nsys-rep']),
        ('nsys_analysis',[sys.executable,str(HERE/'analyze_nsight.py'),str(trace)+'.sqlite'])]
    stages += [('memory_'+name,[sys.executable,'-u',str(HERE/'measure_memory.py'),name]) for name in ['pytorch','cuda','cuda_graph']]
    stages.append(('trajectory',[sys.executable,'-u',str(HERE/'validate_trajectory.py')]))
    status=ROOT/'reports/FINAL_CHECKS_STATUS.json'
    if '--resume' in sys.argv:
        previous=json.loads(status.read_text());assert previous['state']=='failed'
        state['completed']=previous['completed']
    for name,command in stages:
        if name in state['completed']:continue
        state.update(stage=name,started=time.time());status.write_text(json.dumps(state,indent=2))
        with (ROOT/'logs'/('final_'+name+'.log')).open('w') as f:p=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if p.returncode:
            state.update(state='failed',returncode=p.returncode);status.write_text(json.dumps(state,indent=2));raise RuntimeError(name)
        state['completed'].append(name)
    state.update(state='complete',stage=None);status.write_text(json.dumps(state,indent=2))


if __name__=='__main__':main()
