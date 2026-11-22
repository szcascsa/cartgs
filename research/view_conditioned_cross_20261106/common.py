"""本轮独立目录、进程身份与串行执行。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
PYTHON='/home/prmi/anaconda3/envs/zwj/bin/python'
BASE=Path('/home/prmi/zwj/experiments/local_importance_li_20261105')
OLD=Path('/home/prmi/zwj/experiments/view_budget_raw18_20261105')
GROUPS=['selector_view','compensator_view']
SCHEMAS={'selector_view':3,'compensator_view':4}
PREVIOUS=Path('/home/prmi/zwj/experiments/view_conditioned_joint_20261106')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()


def save(p,value):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)


def identity(pid):
    return dict(pid=pid,start_ticks=Path(f'/proc/{pid}/stat').read_text().split()[21],
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def environment():
    env=dict(os.environ);env['OMP_NUM_THREADS']='2'
    env['LD_LIBRARY_PATH']=':'.join([str(ROOT/'runtime/lib'),
        '/home/prmi/zwj/experiments/real_lod_full_joint_20261103/source/third_party/ORB-SLAM3/lib',
        '/home/prmi/zwj/mycartgs/third_party/install/opencv-cuda118-sm86-gcc9/lib',
        '/home/prmi/zwj/mycartgs/third_party/install/jsoncpp-1.9.5/lib',
        '/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/lib','/usr/local/cuda-11.8/lib64'])
    return env


def run(name,command,state,path,output=None):
    started=time.monotonic();last_step=0;progress_time=started
    with (ROOT/'logs'/f'{name}.log').open('w') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=environment(),stdout=log,stderr=subprocess.STDOUT)
        state['active']=dict(name=name,process=identity(child.pid),step=0);save(path,state)
        while child.poll() is None:
            maps=Path(f'/proc/{child.pid}/maps')
            if maps.exists():
                text=maps.read_text()
                if 'libgaussian_mapper.so' in text:
                    assert str(ROOT/'runtime/lib/libgaussian_mapper.so') in text
                    (ROOT/'reports'/f'runtime_maps_{name}.txt').write_text(text)
            if output is not None:
                progress=output/'real_lod_steps.csv'
                if progress.exists():
                    with progress.open('rb') as f:
                        f.seek(max(0,progress.stat().st_size-2048));lines=f.read().decode().splitlines()
                    step=next((int(s.split(',')[0]) for s in reversed(lines) if s.split(',')[0].isdigit()),0)
                    if step!=last_step:last_step=step;progress_time=time.monotonic()
                    state['active']['step']=step
            state['active'].update(seconds=time.monotonic()-started,no_progress_seconds=time.monotonic()-progress_time)
            save(path,state);time.sleep(3)
    assert child.returncode==0,f'{name}退出{child.returncode}，保留现场，检查logs/{name}.log'
    state['active']=None;save(path,state)
    return dict(seconds=time.monotonic()-started,returncode=0)
