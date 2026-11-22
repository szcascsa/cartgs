"""本轮只读checkpoint与独立源码／编译目录。"""
from pathlib import Path
import csv
import hashlib
import json
import os
import sys
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OLD=Path('/home/prmi/zwj/experiments/global_view_joint_lod_20261105')
CHECKPOINT=Path('/home/prmi/zwj/experiments/view_conditioned_li_20261106/runs/full_both_li/23560_shutdown/ply')
DATASET=Path('/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz')
os.environ['TORCH_EXTENSIONS_DIR']=str(ROOT/'build')
os.environ['CUDA_HOME']='/usr/local/cuda-11.8'
os.environ['TORCH_CUDA_ARCH_LIST']='8.6'
os.environ['MAX_JOBS']='2'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['PATH']='/home/prmi/anaconda3/envs/zwj/bin:/usr/local/cuda-11.8/bin:'+os.environ.get('PATH','')
sys.path[:0]=[str(ROOT/'source/python'),str(OLD/'evaluation/python'),str(OLD/'evaluation/python/diff-gaussian-rasterization')]


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):digest.update(b)
    return digest.hexdigest()


def save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False));tmp.replace(path)


def write(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def gpu(label):
    return dict(label=label,unix=time.time(),gpu=subprocess.check_output(['nvidia-smi',
        '--query-gpu=name,uuid,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw','--format=csv,noheader'],text=True),
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True))


def identity():
    pid=os.getpid()
    return dict(pid=pid,start_ticks=Path(f'/proc/{pid}/stat').read_text().split()[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
