"""修复版正式质量测试：固定24模型，旧基线逐view exact gate，无eval调参。"""
from pathlib import Path
import csv, hashlib, importlib.util, json, os, shutil, subprocess, time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'spatial-mlp-implicit-si-20261020'
FIX=ROOT.parent/'spatial-mlp-optimization-fix-20261020'
SCENES=['fr1_room','fr1_xyz','fr2_xyz']; RATIOS=[.01,.05,.1,.15]; SEEDS=[2026,2027]
METRICS=['psnr','ssim','lpips','photo']; ARM='MLP-ST-fixed'
torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
torch.backends.cuda.matmul.allow_tf32=False

def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb')as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def csvread(p):return list(csv.DictReader(open(p)))
def csvwrite(p,rows):
    with open(p,'w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def status(stage,**kw):save(ROOT/'status.json',dict(stage=stage,**kw))

def check_inputs():
    checks=read(OLD/'input_checks.json')['scenes']
    for name,c in checks.items():
        for field in ['input_sha','runtime_sha','legacy_model_sha']:
            for p,h in c[field].items():assert sha(p)==h,(name,p)
        for p,h in c['source_sha'].items():assert sha(p)==h['sha256'],p
        info=read(OLD/name/'input_manifest.json')
        assert sha(info['cameras'])==c['cameras_sha']
        assert not(set(info['bank_frames'])&set(info['evaluation_frames']))
    return dict(all_passed=True,scenes=SCENES)

def renderer(name):
    here=ROOT/name;here.mkdir(exist_ok=True)
    for f in ['common.py','audit_worker','input_manifest.json']:
        destination=here/f
        if not destination.exists():shutil.copy2(OLD/name/f,destination)
        assert sha(destination)==sha(OLD/name/f)
    spec=importlib.util.spec_from_file_location('frozen_'+name,here/'common.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    os.environ['LD_LIBRARY_PATH']=str(OLD/name/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
    return module

def evaluate(baseline):
    allq=[];allv=[];maximum={m:0. for m in METRICS};renders=0
    oldq=csvread(OLD/'quality_summary.csv');oldv=csvread(OLD/'quality_per_view.csv')
    lookup={(z['scene'],z['split'],z['arm'],float(z['ratio']),int(z['seed']),int(z['frame_index'])):z for z in oldv}
    sets=np.load(OLD/'selected_sets.npz' if baseline else ROOT/'selected_sets.npz')
    for name in SCENES:
        module=renderer(name);info=read(ROOT/name/'input_manifest.json')
        scene=module.Scene()
        try:
            metrics=module.Metrics(module.tensors(info['cameras'])['gt'])
            cases=[z for z in oldq if z['scene']==name] if baseline else [dict(scene=name,ratio=r,seed=s,split=split,arm=ARM,step=2000)for r in RATIOS for s in SEEDS for split in ['train','eval']]
            for number,z in enumerate(cases):
                ratio=float(z['ratio']);seed=int(z['seed']);arm=z['arm'];split=z['split']
                key=f'{name}_{arm}_{ratio}'+(f'_{seed}'if seed!=-1 else'')
                rows=sets[key];assert len(rows)==len(np.unique(rows))==int(len(np.load(OLD/name/'ranking_inputs.npz')['row_ids'])*ratio+.5)
                frames=info['bank_frames'if split=='train'else'evaluation_frames'];case=[]
                status('baseline_reproduction'if baseline else'quality',scene=name,arm=arm,ratio=ratio,seed=seed,split=split,case=number,total_cases=len(cases))
                for start in range(0,len(frames),4):
                    indices=frames[start:start+4]
                    response=scene.call(dict(method='images',rows=rows.tolist(),ratio=ratio,frames=indices,apply_tf=True))
                    values=metrics.all(torch.from_numpy(np.array(scene.mapping[:len(indices)])).cuda(),indices)
                    for j,frame in enumerate(indices):
                        row=dict(scene=name,split=split,seed=seed,step=int(z['step']),arm='MLP-ST-legacy'if arm=='MLP-ST'else arm,ratio=ratio,K=len(rows),frame_index=frame,KF_id=info['camera_ids'][frame],photo=response['photo'][j],psnr=float(values[j,0]),ssim=float(values[j,1]),lpips=float(values[j,2]))
                        if baseline:
                            target=lookup[name,split,arm,ratio,seed,frame]
                            for m in METRICS:
                                error=abs(row[m]-float(target[m]));maximum[m]=max(maximum[m],error)
                                assert error==0,('STOP_BASELINE_NOT_EXACT',name,arm,ratio,seed,frame,m,error)
                        case.append(row);renders+=1
                allv.extend(case)
                summary={k:v for k,v in case[0].items()if k not in ['frame_index','KF_id',*METRICS]}
                summary.update(views=len(case),**{m:float(np.mean([a[m]for a in case]))for m in METRICS})
                if baseline:
                    for m in METRICS:assert summary[m]==float(z[m]),('STOP_AGGREGATE_NOT_EXACT',name,arm,m)
                allq.append(summary)
                csvwrite(ROOT/('baseline_quality_summary.csv'if baseline else'fixed_quality_summary.csv'),allq)
                csvwrite(ROOT/('baseline_quality_per_view.csv'if baseline else'fixed_quality_per_view.csv'),allv)
                print('REPRODUCED'if baseline else'QUALITY',name,arm,ratio,seed,split,summary['psnr'],flush=True)
        finally:scene.close()
    save(ROOT/('baseline_reproduction.json'if baseline else'quality_complete.json'),dict(all_passed=True,view_renders=renders,summaries=len(allq),per_view_max_abs=maximum if baseline else None))

def train_and_export():
    curves=[];hard=[];fields=[];rankings={};sets={};checks=[];count=0
    for name in SCENES:
        data=np.load(OLD/name/'ranking_inputs.npz');ids=data['row_ids'];base=np.log(data['raw_I'].astype(float)+1e-12)
        norm=np.load(OLD/name/'normalized_xyz.npz')['normalized_xyz'];raw=np.lexsort((ids,-base))
        np.testing.assert_array_equal(raw,np.load(OLD/name/'rankings.npz')['A_order'])
        for ratio in RATIOS:
            k=int(len(ids)*ratio+.5)
            for seed in SEEDS:
                status('training',scene=name,ratio=ratio,seed=seed,completed=count,total=24)
                out=ROOT/name/'models'/f'fixed_{int(100*ratio)}_seed_{seed}';out.mkdir(parents=True,exist_ok=False)
                env=dict(os.environ,LD_LIBRARY_PATH=str(OLD/name/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
                cmd=[str(FIX/'build/train_worker_fixed'),str(OLD/name/'training_manifest.json'),str(OLD/name/'training_input.bin'),str(out),str(ratio),str(seed),'2000','.0001']
                start=time.time()
                with open(out/'training.log','w')as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
                rec=[json.loads(x[3:])for x in(out/'training.log').read_text().splitlines()if x.startswith('PG ')]
                assert rec[-1]['completed']and rec[-1]['frozen_exact']and rec[-1]['selection_gradient']=='implicit_expected_K_VJP'
                tr=[x for x in rec if x['kind']=='training'];hh=[x for x in rec if x['kind']=='hard_summary'];assert len(tr)==2000
                old=read(OLD/name/'legacy_models'/f'PG_{int(100*ratio)}_seed_{seed}'/'completed.json')['records']
                assert [x['frame']for x in tr]==old[-1]['visited_Bank_frames']
                max_gradient_ratio=max(abs(x['score_grad_sum'])/(x['score_grad_abs_sum']+1e-30)for x in tr)
                assert max_gradient_ratio<1e-12
                for z in tr:
                    assert np.isfinite([z['sampled_photo'],z['theta_grad_norm'],z['correction_std'],z['correction_mean']]).all()
                    for key in ['output_changed_fraction','selected_changed_fraction','unselected_changed_fraction','mean_abs_output_update']:z.setdefault(key,'')
                    layers=z.pop('layer_grad_norm');z.pop('kind');curves.append(dict(scene=name,ratio=ratio,seed=seed,**z,layer1_grad_norm=layers[0],layer2_grad_norm=layers[1],layer3_grad_norm=layers[2]))
                for z in hh:
                    selected=np.array(z.pop('selected_source_rows'));assert len(selected)==len(np.unique(selected))==k
                    sets[f'{name}_{ARM}_{ratio}_{seed}_step_{z["step"]}']=selected;z.pop('kind');hard.append(dict(scene=name,ratio=ratio,seed=seed,**z))
                oldstate=dict(torch.jit.load(str(OLD/name/'legacy_models'/f'PG_{int(100*ratio)}_seed_{seed}'/'step_2000.pt'),map_location='cpu').named_buffers())
                for step in [0,250,1000,2000]:
                    state=dict(torch.jit.load(str(out/f'step_{step}.pt'),map_location='cpu').named_buffers());a=state['correction'].numpy()
                    np.testing.assert_array_equal(state['normalized_xyz'].numpy(),norm)
                    assert torch.equal(state['cached_uniform'],oldstate['cached_uniform'])
                    model=torch.nn.Sequential(torch.nn.Linear(3,32),torch.nn.SiLU(),torch.nn.Linear(32,32),torch.nn.SiLU(),torch.nn.Linear(32,1,bias=False)).double()
                    model.load_state_dict(dict(torch.jit.load(str(out/f'step_{step}_weights.pt'),map_location='cpu').named_parameters()))
                    with torch.no_grad():
                        r=model(state['normalized_xyz']).flatten();torch.testing.assert_close(r-r.mean(),state['correction'],atol=1e-9,rtol=1e-10)
                    if step==0:assert np.all(a==0)
                    score=base+a;order=np.lexsort((ids,-score));selected=np.sort(ids[order[:k]])
                    key=f'{name}_{ARM}_{ratio}_{seed}_step_{step}'
                    np.testing.assert_array_equal(selected,sets[key]);rankings[key]=ids[order]
                    fields.append(dict(scene=name,ratio=ratio,seed=seed,step=step,mean=float(a.mean()),std=float(a.std()),max_abs=float(abs(a).max()),median=float(np.median(a)),p1=float(np.quantile(a,.01)),p99=float(np.quantile(a,.99))))
                    if step==2000:
                        sets[f'{name}_{ARM}_{ratio}_{seed}']=selected
                        np.savez_compressed(out/'final_field.npz',row_ids=ids,normalized_xyz=norm,correction=a,score=score,ranking=ids[order],selected=selected,direct_selected_count=state['direct_selected_count'].numpy())
                shutil.copy2(OLD/name/'normalization.json',out/'normalization.json')
                count+=1;check=dict(scene=name,ratio=ratio,seed=seed,all_passed=True,frozen_exact=True,steps=2000,seconds=time.time()-start,gradient_normalized_sum_max=max_gradient_ratio,uniform_and_Bank_schedule_exact=True,initial_Raw_exact=True,weights_independently_verified=True)
                save(out/'completed.json',check);checks.append(check)
                csvwrite(ROOT/'training_curves.csv',curves);csvwrite(ROOT/'hard_training_dynamics.csv',hard);csvwrite(ROOT/'correction_statistics.csv',fields)
                print('TRAINED',count,name,ratio,seed,flush=True)
    np.savez_compressed(ROOT/'rankings.npz',**rankings);np.savez_compressed(ROOT/'selected_sets.npz',**sets)
    save(ROOT/'selected_before_quality.json',dict(all_passed=True,ranking_sha=sha(ROOT/'rankings.npz'),selected_sha=sha(ROOT/'selected_sets.npz'),models=checks,formal_updates=len(curves)))

def main():
    ROOT.mkdir(exist_ok=True)
    assert read(FIX/'verification.json')['all_correctness_checks_passed']
    assert sha(FIX/'source/train_worker.cpp')==read(FIX/'verification.json')['fixed_source_SHA']
    protocol=dict(scenes=SCENES,ratios=RATIOS,seeds=SEEDS,steps=2000,lr=.0001,architecture='3-32-32-1 SiLU, bias-free final layer',parameters=1216,correction='full population differentiable centering',gradient='implicit expected-K VJP',sampler='legacy cached-uniform ST Bernoulli; expected K',loss='.8 L1+.2(1-fast_ssim)',optimizer='Adam .9/.999 eps1e-8 wd0',final_step=2000,no_eval_tuning=True,no_early_stop=True,previous_100step_sanity='no hard boundary crossing; user now explicitly requests full-budget quality test',source_sha={str(p):sha(p)for p in [FIX/'source/train_worker.cpp',FIX/'source/calibrated_selection.h',FIX/'build/train_worker_fixed',OLD/'quality_summary.csv',OLD/'quality_per_view.csv',OLD/'selected_sets.npz']})
    protocol['metric_environment']=dict(cudnn_benchmark=False,matmul_TF32=False,cudnn_TF32=torch.backends.cudnn.allow_tf32)
    protocol['source_sha'][str(ROOT/'run_quality.py')]=sha(ROOT/'run_quality.py')
    save(ROOT/'preregistration.json',protocol);save(ROOT/'inputs_before.json',check_inputs())
    evaluate(True)
    train_and_export()
    fixed=read(ROOT/'selected_before_quality.json')
    assert sha(ROOT/'selected_sets.npz')==fixed['selected_sha']
    evaluate(False)
    save(ROOT/'inputs_after.json',check_inputs())
    for p,h in protocol['source_sha'].items():assert sha(p)==h,p
    status('complete',models=24,formal_updates=48000)

if __name__=='__main__':
    try:main()
    except Exception as error:
        status('failed',error=str(error));raise
