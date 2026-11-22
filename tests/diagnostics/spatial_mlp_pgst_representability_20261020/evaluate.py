"""只在全部regression结束并锁定集合后，复用原冻结renderer评价PG与MLP-fit。"""
from pathlib import Path
import csv,hashlib,importlib.util,json,os,shutil
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'spatial-mlp-implicit-si-20261020'
SCENES=['fr1_room','fr1_xyz','fr2_xyz'];RATIOS=[.01,.05,.1,.15];SEEDS=[2026,2027];METRICS=['psnr','ssim','lpips','photo']
torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,rows):
    with open(ROOT/name,'w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run():
    fixed=read(ROOT/'fit_complete.json');assert fixed['all_passed']
    assert sha(ROOT/'selected_sets.npz')==fixed['selected_sets_SHA']and sha(ROOT/'rankings.npz')==fixed['rankings_SHA']
    inputs=read(OLD/'input_checks.json')['scenes']
    old=[x for x in csv.DictReader(open(OLD/'quality_per_view.csv'))if x['arm']=='PG-ST']
    lookup={(x['scene'],x['split'],float(x['ratio']),int(x['seed']),int(x['frame_index'])):x for x in old}
    sets=np.load(ROOT/'selected_sets.npz');q=[];views=[];maximum={m:0. for m in METRICS};source_hashes={};render_count=0
    for scene in SCENES:
        c=inputs[scene]
        for p,h in c['source_sha'].items():assert sha(p)==h['sha256']
        for p,h in c['runtime_sha'].items():assert sha(p)==h
        here=ROOT/scene;info=read(OLD/scene/'input_manifest.json');assert not(set(info['bank_frames'])&set(info['evaluation_frames']))
        assert sha(info['cameras'])==c['cameras_sha']
        for f in ['common.py','audit_worker','input_manifest.json']:
            dest=here/f;shutil.copy2(OLD/scene/f,dest);source_hashes[str(dest)]=sha(dest)
        spec=importlib.util.spec_from_file_location('frozen_'+scene,here/'common.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        os.environ['LD_LIBRARY_PATH']=str(OLD/scene/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
        renderer=module.Scene()
        try:
            metrics=module.Metrics(module.tensors(info['cameras'])['gt'])
            for ratio in RATIOS:
                for seed in SEEDS:
                    for arm in ['PG-ST','MLP-fit']:
                        selected=sets[f'{scene}_{ratio}_{seed}_{arm}'];assert len(selected)==len(np.unique(selected))
                        for split in ['train','eval']:
                            frames=info['bank_frames'if split=='train'else'evaluation_frames'];case=[]
                            save(ROOT/'status.json',dict(stage='quality',scene=scene,ratio=ratio,seed=seed,arm=arm,split=split,completed_summaries=len(q)))
                            for start in range(0,len(frames),4):
                                indices=frames[start:start+4];result=renderer.call(dict(method='images',rows=selected.tolist(),ratio=ratio,frames=indices,apply_tf=True))
                                values=metrics.all(torch.from_numpy(np.array(renderer.mapping[:len(indices)])).cuda(),indices)
                                for j,frame in enumerate(indices):
                                    row=dict(scene=scene,ratio=ratio,seed=seed,split=split,arm=arm,K=len(selected),frame_index=frame,KF_id=info['camera_ids'][frame],photo=result['photo'][j],psnr=float(values[j,0]),ssim=float(values[j,1]),lpips=float(values[j,2]))
                                    if arm=='PG-ST':
                                        ref=lookup[scene,split,ratio,seed,frame]
                                        for m in METRICS:
                                            err=abs(row[m]-float(ref[m]));maximum[m]=max(maximum[m],err);assert err==0,('STOP_PGST_NOT_EXACT',scene,ratio,seed,split,frame,m,err)
                                    case.append(row);render_count+=1
                            views.extend(case);summary={k:v for k,v in case[0].items()if k not in ['frame_index','KF_id',*METRICS]}
                            summary.update(views=len(case),**{m:float(np.mean([x[m]for x in case]))for m in METRICS});q.append(summary)
                            write('quality_summary.csv',q);write('quality_per_view.csv',views)
                            print('QUALITY',scene,ratio,seed,arm,split,summary['psnr'],flush=True)
        finally:renderer.close()
        for p,h in c['source_sha'].items():assert sha(p)==h['sha256']
    assert sha(ROOT/'selected_sets.npz')==fixed['selected_sets_SHA']
    save(ROOT/'quality_complete.json',dict(all_passed=True,view_renders=render_count,summaries=len(q),PGST_per_view_max_abs=maximum,source_hashes=source_hashes))
    save(ROOT/'status.json',dict(stage='complete',models=24))

if __name__=='__main__':
    try:run()
    except Exception as error:save(ROOT/'status.json',dict(stage='failed',error=str(error)));raise
