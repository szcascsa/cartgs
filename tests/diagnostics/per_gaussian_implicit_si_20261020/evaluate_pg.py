"""所有ranking先固定，再用旧frozen worker和原四指标评价。"""
import sys,os,csv,json,hashlib
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parent
scene_name=sys.argv[1];HERE=ROOT/scene_name
sys.path.insert(0,str(HERE));from common import Scene,Metrics,tensors
def read(p):return json.loads(Path(p).read_text())
def write(name,rows):
    with open(HERE/name,'w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
reg=read(ROOT/'preregistration.json');fixed=read(ROOT/'selected_before_quality.json')
assert hashlib.sha256((ROOT/'selected_sets.npz').read_bytes()).hexdigest()==fixed['selected_sha']
os.environ['LD_LIBRARY_PATH']=str(HERE/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
sets=dict(np.load(ROOT/'selected_sets.npz'));info=read(HERE/'input_manifest.json');records=[];summary=[]
# Reproduced evaluation baseline is copied; Bank baseline is really rendered below.
for p,target in [('baseline_per_view.csv',records),('baseline_quality.csv',summary)]:
    for z in csv.DictReader(open(HERE/p)):
        if z['arm']not in['A','B','C']:continue
        z={k:(float(v)if k in['ratio','photo','psnr','ssim','lpips']else int(v)if k in['K','views','frame_index','KF_id']else v)for k,v in z.items()}
        target.append(dict(scene=scene_name,split='eval',seed=-1,step=0,**z))
scene=Scene()
try:
    metrics=Metrics(tensors(info['cameras'])['gt'])
    tasks=[]
    for ratio in reg['ratios']:
        for arm in ['A','B','C']:tasks.append((arm,ratio,-1,0,info['bank_frames'],'train',sets[f'{scene_name}_{arm}_{ratio}']))
        for seed in reg['seeds']:
            arm=f'PG-{int(ratio*100)}';case=f'PG_{int(ratio*100)}_seed_{seed}'
            for step in reg['checkpoints']:
                selected=sets[f'{scene_name}_{case}_step_{step}']
                tasks.append((arm,ratio,seed,step,info['bank_frames'],'train',selected))
                if step==reg['steps']:tasks.append((arm,ratio,seed,step,info['evaluation_frames'],'eval',selected))
    for arm,ratio,seed,step,frames,split,selected in tasks:
        case=[]
        (HERE/'quality_status.json').write_text(json.dumps(dict(arm=arm,ratio=ratio,seed=seed,step=step,split=split,completed=len(summary))))
        for start in range(0,len(frames),4):
            indices=frames[start:start+4];out=scene.call(dict(method='images',rows=selected.tolist(),ratio=ratio,frames=indices,apply_tf=True));v=metrics.all(torch.from_numpy(np.array(scene.mapping[:len(indices)])).cuda(),indices)
            for j,f in enumerate(indices):case.append(dict(scene=scene_name,split=split,seed=seed,step=step,arm=arm,ratio=ratio,K=len(selected),frame_index=f,KF_id=info['camera_ids'][f],photo=out['photo'][j],psnr=float(v[j,0]),ssim=float(v[j,1]),lpips=float(v[j,2])))
        records.extend(case);z=dict(scene=scene_name,split=split,seed=seed,step=step,arm=arm,ratio=ratio,K=len(selected),views=len(case),**{m:float(np.mean([a[m]for a in case]))for m in ['photo','psnr','ssim','lpips']});summary.append(z)
        write('pg_quality_summary.csv',summary);write('pg_quality_per_view.csv',records);print('QUALITY',json.dumps(z),flush=True)
finally:scene.close()
(HERE/'pg_quality_complete.json').write_text(json.dumps(dict(all_passed=True,records=len(records),summaries=len(summary))))
