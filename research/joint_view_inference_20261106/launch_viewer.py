"""使用本轮独立扩展和只读Rasterizer依赖启动Viewer；其余参数原样传递。"""
from pathlib import Path
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
OLD=Path('/home/prmi/zwj/experiments/global_view_joint_lod_20261105')
environment=dict(os.environ)
environment['PATH']='/home/prmi/anaconda3/envs/zwj/bin:/usr/local/cuda-11.8/bin:'+environment.get('PATH','')
environment['CUDA_HOME']='/usr/local/cuda-11.8'
environment['TORCH_EXTENSIONS_DIR']=str(ROOT/'build')
environment['TORCH_CUDA_ARCH_LIST']='8.6'
environment['MAX_JOBS']='2'
environment['PYTHONPATH']=os.pathsep.join([str(ROOT/'source/python'),str(OLD/'evaluation/python'),
    str(OLD/'evaluation/python/diff-gaussian-rasterization'),environment.get('PYTHONPATH','')])
if __name__=='__main__':
    raise SystemExit(subprocess.call([sys.executable,str(ROOT/'source/scripts/view_budget.py')]+sys.argv[1:],env=environment))
