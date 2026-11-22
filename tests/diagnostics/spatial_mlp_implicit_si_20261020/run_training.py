"""三值Bank-only LR sanity后锁定统一LR；正式24模型，legacy schedule/uniform exact。"""
from pathlib import Path
import json,subprocess,os,time
import numpy as np,torch
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
reg=read(ROOT/'preregistration.json')
assert read(ROOT/'baseline_all_passed.json')['all_passed']and read(ROOT/'legacy_gate.json')['all_passed']
def execute(name,ratio,seed,steps,lr,out):
    out.mkdir(parents=True,exist_ok=False);start=time.time();env=dict(os.environ,LD_LIBRARY_PATH=str(ROOT/name/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
    cmd=[str(ROOT/'build/train_worker'),str(ROOT/name/'training_manifest.json'),str(ROOT/name/'training_input.bin'),str(out),str(ratio),str(seed),str(steps),str(lr)]
    with open(out/'training.log','w')as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    records=[json.loads(l[3:])for l in(out/'training.log').read_text().splitlines()if l.startswith('PG ')];assert records[-1]['completed']and records[-1]['frozen_exact']
    tr=[z for z in records if z['kind']=='training'];assert len(tr)==steps
    legacy=read(ROOT/name/'legacy_models'/f'PG_{int(100*ratio)}_seed_{seed}'/'completed.json')['records'];frames=legacy[-1]['visited_Bank_frames'];assert [z['frame']for z in tr]==frames[:steps]
    t={k:v.detach()for k,v in torch.jit.load(str(out/'step_0.pt'),map_location='cpu').named_buffers()};old=dict(torch.jit.load(str(ROOT/name/'legacy_models'/f'PG_{int(100*ratio)}_seed_{seed}'/'step_2000.pt'),map_location='cpu').named_buffers())
    assert torch.equal(t['cached_uniform'],old['cached_uniform']);assert torch.all(t['correction']==0)
    n=np.load(ROOT/name/'normalized_xyz.npz');np.testing.assert_array_equal(t['normalized_xyz'].numpy(),n['normalized_xyz'])
    ids=n['row_ids'];base=np.log(np.load(ROOT/name/'ranking_inputs.npz')['raw_I'].astype(float)+1e-12);raw=np.lexsort((ids,-base));np.testing.assert_array_equal(raw,np.lexsort((ids,-(base+t['correction'].numpy()))))
    for ratio_check in reg['ratios']:
        K=int(ratio_check*len(ids)+.5);np.testing.assert_array_equal(ids[raw[:K]],ids[np.lexsort((ids,-(base+t['correction'].numpy())))[:K]])
    for z in tr:assert np.isfinite([z['sampled_photo'],z['theta_grad_norm'],z['correction_std'],z['correction_max_abs']]).all()
    result=dict(all_passed=True,scene=name,ratio=ratio,seed=seed,steps=steps,lr=lr,seconds=time.time()-start,only_theta_optimized=True,legacy_frame_schedule_exact=True,legacy_cached_uniform_exact=True,zero_output_exact=True,Raw_full_ranking_and_four_sets_exact=True,normalized_xyz_exact=True)
    save(out/'completed.json',result);return records,result
s=reg['LR_sanity'];results=[]
for lr in s['lrs']:
    out=ROOT/'lr_sanity'/f'lr_{lr}';records,check=execute(s['scene'],s['ratio'],s['seed'],s['steps'],lr,out)
    h=[z for z in records if z['kind']=='hard_summary'];tr=[z for z in records if z['kind']=='training'];finite=True;updated=max(z['theta_grad_norm']for z in tr)>0 and max(z['correction_std']for z in tr)>0
    results.append(dict(lr=lr,initial_Bank_photo=h[0]['photo'],final_Bank_photo=h[-1]['photo'],photo_delta=h[-1]['photo']-h[0]['photo'],final_Bank_PSNR=h[-1]['psnr'],finite=finite,updated=updated,theta_grad_norm_max=max(z['theta_grad_norm']for z in tr),correction_max_abs=max(z['correction_max_abs']for z in tr),check=check))
valid=[z for z in results if z['finite']and z['updated']];assert valid,'STOP_ALL_LR_FROZEN_OR_NONFINITE';chosen=min(valid,key=lambda z:(z['final_Bank_photo'],z['lr']))
save(ROOT/'lr_sanity.json',dict(all_passed=True,preregistered=s,results=results,selected_lr=chosen['lr'],selection_criterion=s['criterion'],locked_before_formal_training_and_evaluation=True,evaluation_reads=0,baseline_quality_reads=0))
print('LR_LOCKED',chosen['lr'],flush=True)
for name in reg['scenes']:
    for ratio in reg['ratios']:
        for seed in reg['seeds']:
            case=f'MLP_{int(ratio*100)}_seed_{seed}';save(ROOT/'training_status.json',dict(scene=name,ratio=ratio,seed=seed,case=case,stage='training'));out=ROOT/name/'models'/case
            records,check=execute(name,ratio,seed,reg['steps'],chosen['lr'],out);print('TRAINED',name,ratio,seed,round(check['seconds'],2),flush=True)
save(ROOT/'training_complete.json',dict(all_passed=True,models=24,formal_updates=48000,lr=chosen['lr']))
