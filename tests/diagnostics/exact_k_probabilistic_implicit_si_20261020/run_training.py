"""通过概率/梯度/冻结/legacy门控后运行24个Bank-only PL模型。"""
from pathlib import Path
import subprocess,os,json,time
import numpy as np,torch
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
reg=read(ROOT/'preregistration.json')
def execute(scene,ratio,seed,steps,out):
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,LD_LIBRARY_PATH=str(ROOT/scene/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
    cmd=[str(ROOT/'build/train_worker'),str(ROOT/scene/'training_manifest.json'),str(ROOT/scene/'training_input.bin'),str(out),str(ratio),str(seed),str(steps),str(reg['lr'])]
    start=time.time()
    with open(out/'training.log','w')as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    records=[json.loads(l[3:])for l in(out/'training.log').read_text().splitlines()if l.startswith('PG ')]
    assert records[-1]['completed']and records[-1]['frozen_exact']and records[-1]['pure_score_function']
    training=[z for z in records if z['kind']=='training'];assert len(training)==steps
    N=read(ROOT/'input_checks.json')['scenes'][scene]['N'];K=int(ratio*N+.5)
    signatures=set();baselines={z['frame']:z['photo']for z in records if z['kind']=='hard_view'and z['step']==0}
    for i,z in enumerate(training):
        assert z['step']==i+1 and z['sampled_count']==z['K']==K and z['fresh_gumbel']
        sig=tuple(z['noise_signature']);assert sig not in signatures;signatures.add(sig)
        assert z['frame']in baselines and z['baseline_before']==baselines[z['frame']]
        assert z['advantage']==z['train_loss']-z['baseline_before']
        assert z['baseline_after']==.99*z['baseline_before']+.01*z['train_loss']
        assert z['minimum_denominator']>0 and z['minimum_shifted_weight']>0
        baselines[z['frame']]=z['baseline_after']
    save(out/'completed.json',dict(all_passed=True,ratio=ratio,seed=seed,steps=steps,N=N,K=K,seconds=time.time()-start,exact_K_all_steps=True,fresh_all_steps=True,baseline_sequence_exact=True))
    return records
assert read(ROOT/'pl_probability_tests.json')['all_passed']and read(ROOT/'reinforce_gradient_tests.json')['all_passed']
assert read(ROOT/'baseline_all_passed.json')['all_passed']and read(ROOT/'legacy_gate.json')['all_passed']
a=execute('fr1_xyz',.05,2026,5,ROOT/'smoke_first');b=execute('fr1_xyz',.05,2026,5,ROOT/'smoke_repeat')
def tensors(d):return {k:v.detach()for k,v in torch.jit.load(str(ROOT/d/'step_5.pt'),map_location='cpu').named_buffers()}
x=tensors('smoke_first');y=tensors('smoke_repeat');repeat={k:dict(exact=torch.equal(v,y[k]),max_abs=float((v-y[k]).abs().max()))for k,v in x.items()}
N=x['coefficient'].numel();assert x['max_abs_gradient'].min()>0 and(x['gradient_hit_steps']>0).all()
for z in a:
    if z['kind']=='training':assert z['nonzero_grad_N']==N and z['selected_nonzero_grad_N']>0 and z['unselected_nonzero_grad_N']>0
assert all(z['exact']for z in repeat.values()),('STOP_REPEAT_NOT_EXACT',repeat)
# New pure-inference selected renderer must exactly match the old frozen worker.
import sys
sys.path.insert(0,str(ROOT/'fr1_xyz'));import common
os.environ['LD_LIBRARY_PATH']=str(ROOT/'fr1_xyz/runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
scene=common.Scene();inp=np.load(ROOT/'fr1_xyz/ranking_inputs.npz');ids=inp['row_ids'];order=np.lexsort((ids,-inp['raw_I']));K=int(.05*len(ids)+.5);initial=[z for z in a if z['kind']=='hard_view'and z['step']==0]
try:
    for z in initial:
        photo=scene.call(dict(method='images',rows=np.sort(ids[order[:K]]).tolist(),ratio=.05,frames=[z['frame']],apply_tf=True))['photo'][0]
        assert photo==z['photo'],('STOP_RENDER_FORWARD_PARITY',photo,z)
finally:scene.close()
save(ROOT/'training_gate.json',dict(all_passed=True,N=N,all_candidates_selection_gradient=True,selected_and_unselected_gradient=True,frozen_exact=True,renderer_has_no_a_autograd_path=True,pure_score_function=True,all_Bank_forward_photo_exact=True,repeat=repeat))
for name in reg['scenes']:
    for ratio in reg['ratios']:
        for seed in reg['seeds']:
            case=f'PL_{int(100*ratio)}_seed_{seed}';out=ROOT/name/'models'/case
            save(ROOT/'training_status.json',dict(scene=name,ratio=ratio,seed=seed,case=case,stage='training'))
            start=time.time();execute(name,ratio,seed,reg['steps'],out);print('TRAINED',name,ratio,seed,round(time.time()-start,2),flush=True)
save(ROOT/'training_complete.json',dict(all_passed=True,models=24,updates=48000))
