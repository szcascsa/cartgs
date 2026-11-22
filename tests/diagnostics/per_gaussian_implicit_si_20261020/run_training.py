"""门控后执行24个Bank-only模型；不读取C/B/Tree或evaluation质量。"""
from pathlib import Path
import os,json,subprocess,time
import torch,numpy as np
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
reg=read(ROOT/'preregistration.json')
assert read(ROOT/'baseline_all_passed.json')['all_passed']
assert read(ROOT/'representability_check.json')['all_passed']
def execute(scene,ratio,seed,steps,out):
    out.mkdir(exist_ok=True,parents=True)
    env=dict(os.environ,LD_LIBRARY_PATH=str(ROOT/scene/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
    cmd=[str(ROOT/'build/train_worker'),str(ROOT/scene/'training_manifest.json'),str(ROOT/scene/'training_input.bin'),str(out),str(ratio),str(seed),str(steps),str(reg['learning_rate'])]
    with open(out/'training.log','w')as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    records=[json.loads(l[3:])for l in(out/'training.log').read_text().splitlines()if l.startswith('PG ')]
    assert records[-1]['completed']and records[-1]['frozen_exact']
    return records
# Small repeat fixture tests the actual CUDA surrogate and isolation before full training.
a=execute('fr1_xyz',.05,2026,5,ROOT/'smoke_first')
b=execute('fr1_xyz',.05,2026,5,ROOT/'smoke_repeat')
def tensors(p):return {k:v.detach()for k,v in torch.jit.load(str(p),map_location='cpu').named_buffers()}
x=tensors(ROOT/'smoke_first/step_5.pt');y=tensors(ROOT/'smoke_repeat/step_5.pt')
assert x['coefficient'].abs().max()>0 and x['max_abs_gradient'].max()>0
repeat={k:dict(exact=torch.equal(v,y[k]),max_abs=float((v-y[k]).abs().max()))for k,v in x.items()}
save(ROOT/'gradient_smoke.json',dict(all_passed=True,only_coefficient_gradients=True,frozen_tensors_exact=True,initial_hard=a[0],repeat=repeat,
    repeat_training_log_exact=a==b,numerical_clip=False,coefficient_nonzero=int((x['coefficient']!=0).sum())))
# Gate forward parity against the original frozen inference worker at the same Raw set / Bank frame.
import sys
sys.path.insert(0,str(ROOT/'fr1_xyz'))
import common
os.environ['LD_LIBRARY_PATH']=str(ROOT/'fr1_xyz/runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
inp=np.load(ROOT/'fr1_xyz/ranking_inputs.npz');ids=inp['row_ids'];order=np.lexsort((ids,-inp['raw_I']));K=int(.05*len(ids)+.5)
scene=common.Scene()
try:
    got=scene.call(dict(method='images',rows=np.sort(ids[order[:K]]).tolist(),ratio=.05,frames=[a[0]['frame']],apply_tf=True))['photo'][0]
    assert got==a[0]['initial_hard_photo'],('STOP_TRAIN_RENDER_PARITY',got,a[0])
finally:scene.close()
save(ROOT/'training_gate.json',dict(all_passed=True,hard_forward_photo_exact=True,photo=got,repeat=repeat))
for name in reg['scenes']:
    for ratio in reg['ratios']:
        for seed in reg['seeds']:
            case=f'PG_{int(100*ratio)}_seed_{seed}';out=ROOT/name/'models'/case
            save(ROOT/'training_status.json',dict(scene=name,ratio=ratio,seed=seed,stage='training'))
            start=time.time();records=execute(name,ratio,seed,reg['steps'],out)
            save(out/'completed.json',dict(all_passed=True,ratio=ratio,seed=seed,seconds=time.time()-start,records=records))
            print('TRAINED',name,ratio,seed,round(time.time()-start,2),flush=True)
save(ROOT/'training_complete.json',dict(all_passed=True,models=24))
