"""冻结模型Other Head推理消融的固定协议。"""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
PREVIOUS=Path('/home/prmi/zwj/experiments/view_conditioned_li_20261106')
OLD=Path('/home/prmi/zwj/experiments/global_view_joint_lod_20261105')
CHECKPOINT=PREVIOUS/'runs/full_both_li/23560_shutdown/ply'
DATASET=Path('/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz')
TRAIN=[.01,.05,.1,.3,.5,.7]
RATIOS=[.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7]
CASES=[(f'{s}_{r:g}',s,r) for s in ['global','view'] for r in RATIOS]
CASES += [(f'points_{n}','points',n) for n in [1000,2000,3000,4000,5000,10000]]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4194304),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temp.replace(path)
