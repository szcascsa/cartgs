"""只读冻结输入与CPU oracle，模块名避免上游common导入冲突。"""
from pathlib import Path
import sys,os,json,time,math,hashlib,csv,statistics,copy,argparse,subprocess
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
FROZEN=Path('/home/prmi/zwj/experiments/compensator_distillation_replica_all_20261108')
PYTHON=FROZEN/'source/python'
sys.path[:0]=[str(PYTHON),'/home/prmi/zwj/experiments/global_view_joint_lod_20261105/evaluation/python/diff-gaussian-rasterization']
import numpy as np
import torch
from selector import load_selector,slot_eligibility
from gaussian_model import GaussianModel
from compensation_student import load_student,decode_student
from anchor_budget import FixedBudgetRenderer
from anchor_frustum import anchor_frustum_mask,compact_gaussians
from budget_viewer import read_config
from utils import getProjectionMatrix,focal2fov
from selector_sampling import gumbel_selection
from cpu_runtime import CpuRenderer,extension

SCENES=json.loads(Path('/home/prmi/zwj/experiments/sparse_rasterizer_sixhour_20261110_0053/reports/SCENES.json').read_text())
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def rotate(values,i):
    n=i%len(values);value=values[n:]+values[:n];return value[::-1] if i%2 else value

def camera(record):
    from types import SimpleNamespace
    pose=np.eye(4);pose[:3,:3]=np.asarray(record['rotation']);pose[:3,3]=record['position']
    w,h=record['width'],record['height'];fx,fy=focal2fov(record['fx'],w),focal2fov(record['fy'],h)
    view=torch.tensor(np.linalg.inv(pose),dtype=torch.float32).transpose(0,1)
    projection=getProjectionMatrix(.01,100.,fx,fy).transpose(0,1)
    return SimpleNamespace(image_width=w,image_height=h,FoVx=fx,FoVy=fy,znear=.01,zfar=100.,
        world_view_transform=view,projection_matrix=projection,full_proj_transform=view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0),
        camera_center=view.inverse()[3,:3])

@torch.no_grad()
def load_scene(slug):
    spec=next(s for s in SCENES if s['slug']==slug);cp=Path(spec['checkpoint'])
    model=load_selector(str(cp/'selector.pt'),'cpu').eval().requires_grad_(False)
    pc=GaussianModel(3);ply,=list((cp/'point_cloud').glob('iteration_*/point_cloud.ply'));pc.load_ply(str(ply),'cpu')
    for name in ['_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest']:getattr(pc,name).requires_grad_(False)
    config=read_config(cp/'selector_config.yaml');ref=FixedBudgetRenderer(model,pc,config,0,'pytorch')
    records=json.loads((cp/'cameras.json').read_text())
    student_path=FROZEN/'models'/slug/'continuous_calibration/best.pt'
    return dict(spec=spec,checkpoint=cp,model=model,pc=pc,reference=ref,records=records,student_path=student_path,ply=ply)

def call(engine,cam,condition,**kwargs):
    if condition=='full':
        if isinstance(engine,CpuRenderer):
            flat,ids,stats,times=engine.native.prepare(cam.world_view_transform.contiguous(),cam.full_proj_transform.contiguous(),cam.camera_center.contiguous(),
                cam.image_width,cam.image_height,cam.znear,3,1.,0,False,False,False,None,None)
            return engine.packet(flat,len(ids)),ids,dict(candidate_gaussians=len(ids))
        from renderer import prepare_render_data
        ids=torch.arange(engine.total);return prepare_render_data(engine.gaussians),ids,dict(candidate_gaussians=len(ids))
    parameters=dict(view_conditioned=not condition.startswith('global'))
    if condition.startswith('fixed'):parameters['budget']=int(condition[5:])
    elif condition.startswith('global'):parameters['global_ratio']=float(condition[6:])
    else:parameters['ratio_budget']=float(condition[5:])
    if isinstance(engine,CpuRenderer):return engine.prepare(cam,**parameters,**kwargs)
    return engine.prepare(cam,**parameters,measure_time=False)

def cuda_camera(cam):
    value=copy.copy(cam)
    for name in ['world_view_transform','projection_matrix','full_proj_transform','camera_center']:setattr(value,name,getattr(cam,name).cuda())
    return value
def cuda_packet(packet):return dict(sh_degree=packet['sh_degree'],inputs={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in packet['inputs'].items()})
