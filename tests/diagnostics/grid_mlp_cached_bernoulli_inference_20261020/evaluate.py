"""Cached推理真实render；同L对照、原hard-K历史结果与逐view exact Raw gate。"""
from pathlib import Path
import csv,hashlib,importlib.util,json,os,shutil
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent;GRID=ROOT.parent/'grid-mlp-implicit-si-20261020';OLD=ROOT.parent/'spatial-mlp-implicit-si-20261020'
SCENES=['fr1_room','fr1_xyz','fr2_xyz'];METRICS=['psnr','ssim','lpips','photo']
torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
read=lambda p:json.loads(p.read_text());load=lambda p:list(csv.DictReader(open(p)))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,z):(ROOT/name).write_text(json.dumps(z,indent=2,allow_nan=False)+'\n')
def write(name,rows):
    with open(ROOT/name,'w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def check_inputs():
    for scene,c in read(OLD/'input_checks.json')['scenes'].items():
        for path,z in c['source_sha'].items():assert sha(Path(path))==z['sha256'],path
        for path,h in c['runtime_sha'].items():assert sha(Path(path))==h,path
        info=read(OLD/scene/'input_manifest.json');assert sha(Path(info['cameras']))==c['cameras_sha']
        assert not set(info['bank_frames'])&set(info['evaluation_frames'])
    return True
def renderer(scene):
    here=ROOT/scene
    for name in ['common.py','audit_worker','input_manifest.json']:shutil.copy2(OLD/scene/name,here/name)
    spec=importlib.util.spec_from_file_location('cached_frozen_'+scene,here/'common.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    os.environ['LD_LIBRARY_PATH']=str(OLD/scene/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
    return module

def run():
    locked=read(ROOT/'selected_before_quality.json');assert locked['all_passed']and locked['checkpoints_unchanged']
    assert sha(ROOT/'selected_sets.npz')==locked['selected_SHA']and check_inputs()
    assert all(sha(Path(path))==h for path,h in locked['model_SHA'].items())
    manifest=read(GRID/'artifact_manifest.json')['files']
    q=[];v=[]
    for filename,target in [('quality_summary.csv',q),('quality_per_view.csv',v)]:
        assert sha(GRID/filename)==manifest[filename]['sha256']
        for z in load(GRID/filename):
            values={key:float(value)if key in ['ratio',*METRICS]else int(value)if key in ['seed','step','K','views','frame_index','KF_id']else value for key,value in z.items()}
            item=dict(scene=values['scene'],split=values['split'],variant=values['arm']+'-Hard-K',ratio=values['ratio'],seed=values['seed'],K=values['K'],L=values['K'],step=values['step'])
            if 'views'in values:item['views']=values['views']
            else:item.update(frame_index=values['frame_index'],KF_id=values['KF_id'])
            item.update({metric:values[metric]for metric in METRICS});target.append(item)
    raw_reference={(z['scene'],z['split'],z['ratio'],z['frame_index']):z for z in v if z['variant']=='Raw-Hard-K'}
    sets=np.load(ROOT/'selected_sets.npz');oldsets=np.load(GRID/'selected_sets.npz');cases=load(ROOT/'evaluation_cases.csv');renders=gate_renders=completed=0
    for phase in ['Raw_gate','cached_quality']:
        for name in SCENES:
            module=renderer(name);info=read(ROOT/name/'input_manifest.json');scene=module.Scene()
            try:
                metrics=module.Metrics(module.tensors(info['cameras'])['gt'])
                own=[dict(variant='Raw-Hard-K',ratio=r,seed=-1,K=int(len(np.load(OLD/name/'ranking_inputs.npz')['row_ids'])*r+.5),L=int(len(np.load(OLD/name/'ranking_inputs.npz')['row_ids'])*r+.5),selected_key=f'{name}_Raw_{r}')for r in [.01,.05,.1,.15]]if phase=='Raw_gate'else[z for z in cases if z['scene']==name]
                for case in own:
                    ratio=float(case['ratio']);seed=int(case['seed']);variant=case['variant'];k=int(case['K']);L=int(case['L']);selected=(oldsets if phase=='Raw_gate'else sets)[case['selected_key']]
                    assert len(selected)==len(np.unique(selected))==L
                    for split in ['train','eval']:
                        save('status.json',dict(stage=phase,scene=name,variant=variant,ratio=ratio,seed=seed,split=split,completed_cases=completed,total_cases=168))
                        frames=info['bank_frames'if split=='train'else'evaluation_frames'];rows=[]
                        for start in range(0,len(frames),4):
                            indices=frames[start:start+4];result=scene.call(dict(method='images',rows=selected.tolist(),ratio=ratio,frames=indices,apply_tf=True));values=metrics.all(torch.from_numpy(np.array(scene.mapping[:len(indices)])).cuda(),indices)
                            for j,frame in enumerate(indices):
                                z=dict(scene=name,split=split,variant=variant,ratio=ratio,seed=seed,K=k,L=L,step=0 if variant in ['Raw-Hard-K','Raw-Cached']else 2000,frame_index=frame,KF_id=info['camera_ids'][frame],psnr=float(values[j,0]),ssim=float(values[j,1]),lpips=float(values[j,2]),photo=result['photo'][j])
                                if phase=='Raw_gate':
                                    for metric in METRICS:assert z[metric]==raw_reference[name,split,ratio,frame][metric],('STOP_RAW_GATE',name,ratio,frame,metric)
                                    gate_renders+=1
                                else:rows.append(z);renders+=1
                        if phase!='Raw_gate':
                            v.extend(rows);z={key:value for key,value in rows[0].items()if key not in ['frame_index','KF_id',*METRICS]};z.update(views=len(rows),**{metric:float(np.mean([r[metric]for r in rows]))for metric in METRICS});q.append(z)
                            write('quality_summary.csv',q);write('quality_per_view.csv',v)
                        print('QUALITY',name,variant,ratio,seed,split,flush=True)
                    if phase!='Raw_gate':completed+=1
            finally:scene.close()
        if phase=='Raw_gate':save('baseline_reproduction.json',dict(all_passed=True,renders=gate_renders,per_view_all_four_exact=True,baseline_file_SHA={name:sha(GRID/name)for name in ['quality_summary.csv','quality_per_view.csv']}))
    assert check_inputs()and sha(ROOT/'selected_sets.npz')==locked['selected_SHA']and sha(ROOT/'inference_arrays.npz')==locked['arrays_SHA']
    assert all(sha(Path(path))==h for path,h in locked['model_SHA'].items())
    assert completed==168 and renders==19880 and gate_renders==1420
    save('quality_complete.json',dict(all_passed=True,new_view_renders=renders,Raw_gate_renders=gate_renders,models_unchanged=True,nominal_ratio_to_TF_unchanged=True))
    save('status.json',dict(stage='quality_complete',completed_cases=completed))

if __name__=='__main__':
    try:run()
    except Exception as error:save('status.json',dict(stage='failed',error=str(error)));raise
