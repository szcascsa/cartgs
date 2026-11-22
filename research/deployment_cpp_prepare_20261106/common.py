"""独立GPU实验路径、真实checkpoint和逐bit比较。"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OLD=Path('/home/prmi/zwj/experiments/global_view_joint_lod_20261105')
os.environ.setdefault('TORCH_EXTENSIONS_DIR',str(ROOT/'build'))
os.environ.setdefault('CUDA_HOME','/usr/local/cuda-11.8')
os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.6')
os.environ.setdefault('MAX_JOBS','2')
os.environ['PATH']='/home/prmi/anaconda3/envs/zwj/bin:/usr/local/cuda-11.8/bin:'+os.environ.get('PATH','')
sys.path[:0]=[str(ROOT/'source/python'),str(OLD/'evaluation/python'),str(OLD/'evaluation/python/diff-gaussian-rasterization')]
import torch
import numpy as np
from budget_viewer import BudgetScene
from frozen_inference_cuda import FrozenResidualInference
from anchor_budget import FixedBudgetRenderer
from anchor_frustum import compact_gaussians,anchor_frustum_mask
from renderer import render_prepared,prepare_render_data
from utils import MiniCam,focal2fov
from selector import compensated_attributes

torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False
DATASET='/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz'
CHECKPOINTS={g:Path('/home/prmi/zwj/experiments/view_conditioned_li_20261106/runs')/('full_'+g)/'23560_shutdown/ply'
             for g in ['both_li','selector_li','compensator_li']}
CHECKPOINTS['mode']=Path('/home/prmi/zwj/experiments/view_conditioned_joint_20261106/runs/full_mode/23560_shutdown/ply')
CHECKPOINTS['legacy']=Path('/home/prmi/zwj/experiments/local_importance_li_20261105/runs/full_no_li/23560_shutdown/ply')
RATIOS=[.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7]


def save(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False));temporary.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scene(group='both_li'):
    return BudgetScene(CHECKPOINTS[group],DATASET,5000,0,None,'pytorch')


def camera(scene,frame):
    c=scene.calibration;v=scene.views[frame]
    return MiniCam(c['width'],c['height'],focal2fov(c['fx'],c['width']),focal2fov(c['fy'],c['height']),np.linalg.inv(np.asarray(v['pose'])))


def reference_renderer(s, graph=False):
    return FixedBudgetRenderer(s.model,s.gaussians,s.config,0,'cuda_graph' if graph else 'cuda')


class Comparisons:
    def __init__(self,name):self.path=ROOT/'reports'/name;self.checks=[]
    def same(self,name,a,b):
        a,b=[v.detach().clone(memory_format=torch.contiguous_format) for v in (a,b)]
        ok=a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.reshape(-1).view(torch.uint8),b.reshape(-1).view(torch.uint8))
        row=dict(name=name,shape=list(a.shape),bitwise=ok)
        if not ok and a.shape==b.shape:
            row.update(max_abs=float((a-b).abs().max()) if a.numel() else 0,
                       different=int((a.reshape(-1).view(torch.uint8)!=b.reshape(-1).view(torch.uint8)).sum()))
        self.checks.append(row)
        if not ok:self.finish(False);raise AssertionError(row)
    def packet(self,name,a,b):
        assert a['sh_degree']==b['sh_degree']
        for key,value in b['inputs'].items():
            if value is None:assert a['inputs'][key] is None
            else:self.same(name+'/'+key,a['inputs'][key],value)
    def finish(self,passed=True,**metadata):
        save(self.path,dict(passed=passed,comparisons=len(self.checks),checks=self.checks,**metadata))
