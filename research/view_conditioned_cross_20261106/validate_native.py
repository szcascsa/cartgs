"""原生边界/梯度检查和两组36步验收，禁止覆盖已有输出。"""
import json
from common import ROOT,BASE,GROUPS,SCHEMAS,sha,save,identity,run
import os
import subprocess
import time
from common import environment


def main():
    path=ROOT/'reports/VALIDATION_STATUS.json';assert not path.exists()
    test_root=ROOT/'runs'/('validation_'+str(time.time_ns()));test_root.mkdir(exist_ok=False)
    state=dict(state='native_tests',controller=identity(os.getpid()),tests=[],smoke={},test_root=str(test_root));save(path,state)
    import torch
    for version in ['old','new']:
        env=environment()
        if version=='old':env['LD_LIBRARY_PATH']=str(BASE/'runtime/lib')+':'+env['LD_LIBRARY_PATH']
        with (ROOT/'logs'/f'cuda_parity_{version}.log').open('w') as log:
            subprocess.run([str(ROOT/'runtime/bin/joint_view_cuda_parity_test'),str(ROOT/'runs'/f'cuda_parity_{version}.pt')],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    a=torch.jit.load(str(ROOT/'runs/cuda_parity_old.pt'),map_location='cpu').state_dict()
    b=torch.jit.load(str(ROOT/'runs/cuda_parity_new.pt'),map_location='cpu').state_dict()
    assert a.keys()==b.keys()
    assert all(torch.equal(v.contiguous().reshape(-1).view(torch.uint8),b[k].contiguous().reshape(-1).view(torch.uint8)) for k,v in a.items())
    state['tests'].append(dict(name='legacy_cuda_old_new_bitwise',cases=10,tensors=len(a),passed=True));save(path,state)
    tests=[('joint_'+m,'joint_view_conditioning_test',[m,str(test_root/('test_joint_'+m))]) for m in ['unit','cross_unit','mode','view','selector_view','compensator_view','mode_empty','view_empty','selector_view_empty','compensator_view_empty']]
    tests += [('selector19_'+m,'view_input_test',[m,str(test_root/('test_selector19_'+m+'.pt'))]) for m in ['cpu','cuda']]
    tests += [('old_budget_'+m,'view_budget_training_test',[m,str(test_root/('test_old_budget_'+m))]) for m in ['sampling','joint','empty','health']]
    tests += [('old_residual_'+m,'view_budget_residual_test',[m,str(test_root/('test_old_residual_'+m))]) for m in ['unit','raw_unit','no_li','raw_no_li']]
    tests += [('old_li_'+m,'local_importance_test',[m,str(test_root/('test_old_li_'+m))]) for m in ['unit','isolation','on','off']]
    for name,binary,args in tests:
        state['tests'].append(dict(name=name,**run(name,[str(ROOT/'runtime/bin'/binary)]+args,state,path)));save(path,state)
    state['state']='checkpoint_smoke';save(path,state)
    for group in GROUPS:
        manifest=ROOT/'configs'/f'{group}.json';value=json.loads(manifest.read_text());output=__import__('pathlib').Path(value['output']);assert not output.exists()
        state['smoke'][group]=run('smoke_'+group,[str(ROOT/'runtime/bin/lod_from_full'),str(manifest),'36'],state,path,output)
        assert json.loads((output/'training_verified.json').read_text())['passed']
        fine=json.loads((output/'12016_shutdown/ply/real_lod_state.json').read_text())
        for key in ['full_training_renders','lod_training_renders','view_budget_training_renders','optimizer_commits','master_updates','selector_updates']:assert fine[key]==36,key
        assert fine['view_budget_conditioning']==SCHEMAS[group]
        assert not fine['view_budget_view_residual'] and fine['view_budget_li_weight']==0 and fine['view_budget_li_backwards']==0
        assert fine['ratio_steps']==fine['view_budget_ratio_steps']==[6]*6
        anchor=json.loads((output/'12016_shutdown/ply/point_cloud/iteration_12016/anchor_selector_state.json').read_text())
        assert anchor['gaussian_count']==13298 and anchor['anchor_count']==2790
        assert [s['adam_step'] for s in anchor['compensation_feature_state']]==[36,30,24,18,12,6]
    import torch
    reference=torch.jit.load(str(BASE/'runs/full_no_li/step0_selector.pt'),map_location='cpu').state_dict()
    first=None
    for group in GROUPS:
        folder=ROOT/'runs'/f'smoke_{group}';values=torch.jit.load(str(folder/'step0_selector.pt'),map_location='cpu').state_dict()
        assert len(reference)==61 and len(values)==67
        for name,a in reference.items():
            b=values[name][:,:11] if name=='local_emd.0.weight' else values[name]
            assert a.shape==b.shape and torch.equal(a.reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8)),name
        assert sha(folder/'step0_master.ply')==sha(BASE/'runs/full_no_li/step0_master.ply')
        assert (folder/'training_keyframes.csv').read_text().splitlines()==(BASE/'runs/full_no_li/training_keyframes.csv').read_text().splitlines()[:37]
        if first is None:first=values
        else:assert all(torch.equal(a,values[n]) for n,a in first.items() if n!='joint_view_conditioning_schema')
    state.update(state='native_smoke_complete',active=None,original_tensor_values_bitwise=61,added_state_tensors=6,smoke_steps=36);save(path,state)


if __name__=='__main__':
    try:main()
    except Exception as error:
        p=ROOT/'reports/VALIDATION_STATUS.json';v=json.loads(p.read_text());v.update(state='failed',error=repr(error));save(p,v);raise
