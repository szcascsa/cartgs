"""锁定cell模型集合后做真实same-K评价；旧五基线复用，Raw重新render exact gate。"""
from pathlib import Path
import csv,hashlib,importlib.util,json,os,shutil
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'spatial-mlp-implicit-si-20261020';PRIOR=ROOT.parent/'spatial-mlp-calibrated-quality-20261020'
SCENES=['fr1_room','fr1_xyz','fr2_xyz'];RATIOS=[.01,.05,.1,.15];SEEDS=[2026,2027];METRICS=['psnr','ssim','lpips','photo']
torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return list(csv.DictReader(open(p)))
def write(name,rows):
    with open(ROOT/name,'w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def renderer(name):
    here=ROOT/name
    for filename in ['common.py','audit_worker','input_manifest.json']:shutil.copy2(OLD/name/filename,here/filename)
    spec=importlib.util.spec_from_file_location('grid_frozen_'+name,here/'common.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    os.environ['LD_LIBRARY_PATH']=str(OLD/name/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
    return module
def check_inputs():
    for scene,c in read(OLD/'input_checks.json')['scenes'].items():
        for path,h in c['source_sha'].items():assert sha(path)==h['sha256'],path
        for path,h in c['runtime_sha'].items():assert sha(path)==h,path
        info=read(OLD/scene/'input_manifest.json');assert sha(info['cameras'])==c['cameras_sha']
        assert not(set(info['bank_frames'])&set(info['evaluation_frames']))
    return True
def run():
    locked=read(ROOT/'selected_before_quality.json');assert locked['all_passed']and sha(ROOT/'selected_sets.npz')==locked['selected_SHA']
    assert check_inputs()
    assert read(PRIOR/'correctness.json')['all_passed']
    manifest=read(PRIOR/'artifact_manifest.json')['files']
    for filename in ['quality_summary.csv','quality_per_view.csv']:assert sha(PRIOR/filename)==manifest[filename]['sha256']
    q=[];v=[]
    for filename,target in [('quality_summary.csv',q),('quality_per_view.csv',v)]:
        for z in load(PRIOR/filename):
            if z['arm']not in ['Raw','PG-ST','Fixed','C','MLP-ST-fixed']:continue
            if z['arm']=='MLP-ST-fixed':z['arm']='Per-Gaussian-MLP'
            target.append({k:float(value)if k in ['ratio',*METRICS]else int(value)if k in ['seed','step','K','views','frame_index','KF_id']else value for k,value in z.items()})
    reference={(z['scene'],z['split'],float(z['ratio']),int(z['frame_index'])):z for z in v if z['arm']=='Raw'}
    sets=np.load(ROOT/'selected_sets.npz');max_error={m:0. for m in METRICS};renders=0;rawrenders=0
    for phase in ['Raw_gate','model_quality']:
        for name in SCENES:
            module=renderer(name);info=read(ROOT/name/'input_manifest.json');scene=module.Scene()
            try:
                metrics=module.Metrics(module.tensors(info['cameras'])['gt'])
                cases=[('Raw',ratio,-1)for ratio in RATIOS]if phase=='Raw_gate'else[(arm,ratio,seed)for arm in ['Grid-MLP','Independent-Cell']for ratio in RATIOS for seed in SEEDS]
                for arm,ratio,seed in cases:
                    key=f'{name}_Raw_{ratio}'if arm=='Raw'else f'{name}_{arm}_{ratio}_{seed}';selected=sets[key]
                    assert len(selected)==len(np.unique(selected))
                    for split in ['train','eval']:
                        save(ROOT/'status.json',dict(stage=phase,scene=name,arm=arm,ratio=ratio,seed=seed,split=split))
                        frames=info['bank_frames'if split=='train'else'evaluation_frames'];case=[]
                        for start in range(0,len(frames),4):
                            indices=frames[start:start+4];result=scene.call(dict(method='images',rows=selected.tolist(),ratio=ratio,frames=indices,apply_tf=True));values=metrics.all(torch.from_numpy(np.array(scene.mapping[:len(indices)])).cuda(),indices)
                            for j,frame in enumerate(indices):
                                z=dict(scene=name,split=split,seed=seed,step=0 if arm=='Raw'else 2000,arm=arm,ratio=ratio,K=len(selected),frame_index=frame,KF_id=info['camera_ids'][frame],photo=result['photo'][j],psnr=float(values[j,0]),ssim=float(values[j,1]),lpips=float(values[j,2]))
                                if arm=='Raw':
                                    for metric in METRICS:
                                        error=abs(z[metric]-float(reference[name,split,ratio,frame][metric]));max_error[metric]=max(max_error[metric],error);assert error==0,('STOP_RAW_BASELINE_NOT_EXACT',name,ratio,frame,metric,error)
                                    rawrenders+=1
                                else:case.append(z);renders+=1
                        if arm!='Raw':
                            v.extend(case);summary={k:value for k,value in case[0].items()if k not in ['frame_index','KF_id',*METRICS]}
                            summary.update(views=len(case),**{metric:float(np.mean([z[metric]for z in case]))for metric in METRICS});q.append(summary)
                            write('quality_summary.csv',q);write('quality_per_view.csv',v)
                        print('QUALITY',name,arm,ratio,seed,split,flush=True)
            finally:scene.close()
        if phase=='Raw_gate':save(ROOT/'baseline_reproduction.json',dict(all_passed=True,raw_view_renders=rawrenders,per_view_max_abs=max_error,reused_baselines=['Raw','PG-ST','Fixed','C','Per-Gaussian-MLP'],prior_source_SHA={name:sha(PRIOR/name)for name in ['quality_summary.csv','quality_per_view.csv']}))
    assert check_inputs()and sha(ROOT/'selected_sets.npz')==locked['selected_SHA']
    save(ROOT/'quality_complete.json',dict(all_passed=True,new_model_view_renders=renders,Raw_check_renders=rawrenders,baseline_files_SHA_unchanged=True))
    save(ROOT/'status.json',dict(stage='quality_complete'))

if __name__=='__main__':
    try:run()
    except Exception as error:save(ROOT/'status.json',dict(stage='failed',error=str(error)));raise
