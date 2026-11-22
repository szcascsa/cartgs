"""独立进程核对历史/新四Head的初始权重与Bank帧序，记录同seed非确定性。"""
import csv
import json
from pathlib import Path
import subprocess
import sys
import torch

ROOT=Path(__file__).resolve().parent
OLD=Path('/home/prmi/zwj/s00-raw-shared-split-20261026')
PYTHON=sys.executable
script="""import torch
import run
torch.manual_seed(3100);torch.cuda.manual_seed_all(3100)
scene=run.rt.Scene('fr1_xyz')
cfg=CFG
torch.manual_seed(3100);torch.cuda.manual_seed_all(3100)
net=run.Network(scene,cfg,3100)
torch.save(dict(state_dict=net.state_dict(),schedule=run.rt.schedule(scene,3100,3000)),PATH)
"""
common=dict(id='RawSharedE16Split4',shared=0,ratio=0,extra=16,selector_depth=1,compensation_depth=2,
    compensation_input='raw11',attach_shared=True)
for name,path,cfg in [('old',OLD,dict(common,split_heads=True)),('new',ROOT,dict(common,head_kind='split4'))]:
    code=script.replace('CFG',repr(cfg)).replace('PATH',repr(str(ROOT/('initial_'+name+'.pt'))))
    subprocess.run([PYTHON,'-c',code],cwd=path,check=True,stdout=subprocess.DEVNULL)
a=torch.load(ROOT/'initial_old.pt',map_location='cpu')
b=torch.load(ROOT/'initial_new.pt',map_location='cpu')
assert a['state_dict'].keys()==b['state_dict'].keys()
differences={k:float((a['state_dict'][k]-b['state_dict'][k]).abs().max()) for k in a['state_dict']}
assert a['schedule']==b['schedule']
assert all(value==0 for value in differences.values())
result=dict(initial_state_dict_exact=True,maximum_initial_parameter_error=max(differences.values()),
    schedule_exact=True,seed=3100,
    note='identical initialization/schedule do not ensure deterministic CUDA renderer backward or complete training')
def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))
old_curve=read(OLD/'results/fr1_xyz/RawSharedE16Split4/3100/training_curve.csv')
new_curve=read(ROOT/'results/fr1_xyz/RawSharedE16Split4/3100/training_curve.csv')
assert [(r['step'],r['ratio'],r['view']) for r in old_curve]==[(r['step'],r['ratio'],r['view']) for r in new_curve]
result['logged_frame_ratio_step_sequence_exact']=True
result['initial_logged_training_rows']=[dict(step=x['step'],old_photo=float(x['photo']),new_photo=float(y['photo']),
    photo_delta=float(y['photo'])-float(x['photo']),old_selected=int(x['selected']),new_selected=int(y['selected']))
    for x,y in zip(old_curve[:8],new_curve[:8])]
(ROOT/'repeat_initialization.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
