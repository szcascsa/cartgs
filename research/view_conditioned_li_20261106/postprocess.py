"""三组完整训练结束后串行评价和汇总，不与训练并发。"""
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import torch
from common import ROOT,HERE,BASE,GROUPS,SCHEMAS,PYTHON,identity,save


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
        assert bool(selector[:,:7].count_nonzero())==(g in ['selector_li','both_li'])
        assert bool(compensation[:,:7].count_nonzero())==(g in ['compensator_li','both_li'])
        first=json.loads((folder/'view_budget_steps.jsonl').read_text().splitlines()[0])
        rows=[json.loads(s) for s in (folder/'view_budget_steps.jsonl').read_text().splitlines()]
        assert len(rows)==11580
        for row in rows:
            assert row['li_supervised'] and row['li_teacher_K']==min(math.floor(row['ratio']*row['candidate_N']),row['eligible_N'])
        fields=['keyframe_id','ratio','K','global_K','photo_loss','budget_loss','full_loss','global_loss']
        comparisons={k:dict(reference=reference[k],actual=first[k],equal=reference[k]==first[k]) for k in fields}
        assert all(v['equal'] for v in comparisons.values()),'首步原始Loss或采样K与无LI起点不匹配'
        assert first['li_supervised'] and first['li_loss']>=0 and abs(first['li_weight']-.1)<1e-6
        audit.append(dict(group=g,selector_view_norm=float(selector[:,:7].norm()),selector_mode_norm=float(selector[:,7].norm()),
            compensation_view_norm=float(compensation[:,:7].norm()),compensation_mode_norm=float(compensation[:,7].norm()),
            first_step=comparisons,initial_li_loss=first['li_loss'],initial_joint_loss=first['joint_loss']))
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
