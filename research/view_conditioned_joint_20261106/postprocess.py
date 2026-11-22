"""两组训练全部结束后，串行评价并汇总；不与训练并发。"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import torch
from common import ROOT,HERE,BASE,GROUPS,PYTHON,identity,save


def main():
    lock=(ROOT/'reports/postprocess.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((ROOT/'reports/TRAINING_STATUS.json').read_text())['state']=='training_complete'
    audit=[]
    reference=json.loads((BASE/'runs/full_no_li/view_budget_steps.jsonl').read_text().splitlines()[0])
    for g in GROUPS:
        folder=ROOT/'runs'/f'full_{g}'
        state=torch.jit.load(str(folder/'23560_shutdown/ply/selector.pt'),map_location='cpu').state_dict()
        selector=state['local_emd.0.weight'][:,11:]
        compensation=state['spatial_compensation.view_projection.weight']
        if g=='mode':assert selector[:,:7].count_nonzero()==0 and compensation[:,:7].count_nonzero()==0
        else:assert selector[:,:7].count_nonzero()>0 and compensation[:,:7].count_nonzero()>0
        first=json.loads((folder/'view_budget_steps.jsonl').read_text().splitlines()[0])
        fields=['keyframe_id','ratio','K','global_K','photo_loss','budget_loss','full_loss','global_loss','joint_loss']
        comparisons={k:dict(reference=reference[k],actual=first[k],equal=reference[k]==first[k]) for k in fields}
        audit.append(dict(group=g,selector_view_norm=float(selector[:,:7].norm()),selector_mode_norm=float(selector[:,7].norm()),
            compensation_view_norm=float(compensation[:,:7].norm()),compensation_mode_norm=float(compensation[:,7].norm()),
            first_step=comparisons))
    save(ROOT/'reports/CONDITION_PARAMETER_AUDIT.json',audit)
    state=dict(controller=identity(os.getpid()),state='running')
    for name in ['evaluate_full.py','summarize.py']:
        state['stage']=name;save(ROOT/'reports/POSTPROCESS.json',state)
        env=dict(os.environ);env['PYTHONDONTWRITEBYTECODE']='1';env['OMP_NUM_THREADS']='2'
        with (ROOT/'logs'/f'postprocess_{name}.log').open('w') as f:
            result=subprocess.run([PYTHON,'-u',str(HERE/name)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:
            state.update(state='failed',returncode=result.returncode);save(ROOT/'reports/POSTPROCESS.json',state);raise RuntimeError(name+'失败，检查日志')
    state['state']='metrics_complete';save(ROOT/'reports/POSTPROCESS.json',state)


if __name__=='__main__':main()
