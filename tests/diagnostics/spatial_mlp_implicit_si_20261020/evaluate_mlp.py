"""固定score导出后用原frozen renderer/metric评价Bank和non-Bank。"""
from pathlib import Path
import sys,os,csv,json,hashlib
import numpy as np,torch
ROOT=Path(__file__).resolve().parent;name=sys.argv[1];HERE=ROOT/name
sys.path.insert(0,str(HERE));from common import Scene,Metrics,tensors
reg=json.loads((ROOT/'preregistration.json').read_text());fixed=json.loads((ROOT/'selected_before_quality.json').read_text());assert hashlib.sha256((ROOT/'selected_sets.npz').read_bytes()).hexdigest()==fixed['selected_sha']
torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
os.environ['LD_LIBRARY_PATH']=str(HERE/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
info=json.loads((HERE/'input_manifest.json').read_text());sets=np.load(ROOT/'selected_sets.npz');records=[];summary=[]
def write(n,rows):
    with open(HERE/n,'w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for n,target in [('legacy_quality_summary.csv',summary),('legacy_quality_per_view.csv',records)]:
    for z in csv.DictReader(open(ROOT/n)):
        if z['scene']!=name:continue
        if z['arm'].startswith('PG-'):
            if int(z['step'])!=2000:continue
            z['arm']='PG-ST'
        elif z['arm']in ['A','B','C']:z['arm']=dict(A='Raw',B='Fixed',C='C')[z['arm']]
        else:continue
        z={k:float(v)if k in['ratio','psnr','ssim','lpips','photo']else int(v)if k in['seed','step','K','views','frame_index','KF_id']else v for k,v in z.items()};target.append(z)
scene=Scene()
try:
    metrics=Metrics(tensors(info['cameras'])['gt'])
    for ratio in reg['ratios']:
        for seed in reg['seeds']:
            selected=sets[f'{name}_MLP-ST_{ratio}_{seed}']
            for split in ['train','eval']:
                case=[];frames=info['bank_frames'if split=='train'else'evaluation_frames']
                (HERE/'quality_status.json').write_text(json.dumps(dict(ratio=ratio,seed=seed,split=split,completed=len(summary))))
                for start in range(0,len(frames),4):
                    indices=frames[start:start+4];out=scene.call(dict(method='images',rows=selected.tolist(),ratio=ratio,frames=indices,apply_tf=True));values=metrics.all(torch.from_numpy(np.array(scene.mapping[:len(indices)])).cuda(),indices)
                    for j,f in enumerate(indices):case.append(dict(scene=name,split=split,seed=seed,step=2000,arm='MLP-ST',ratio=ratio,K=len(selected),frame_index=f,KF_id=info['camera_ids'][f],photo=out['photo'][j],psnr=float(values[j,0]),ssim=float(values[j,1]),lpips=float(values[j,2])))
                records.extend(case);z=dict(scene=name,split=split,seed=seed,step=2000,arm='MLP-ST',ratio=ratio,K=len(selected),views=len(case),**{m:float(np.mean([a[m]for a in case]))for m in ['photo','psnr','ssim','lpips']});summary.append(z)
                write('quality_summary.csv',summary);write('quality_per_view.csv',records);print('QUALITY',json.dumps(z),flush=True)
finally:scene.close()
(HERE/'quality_complete.json').write_text(json.dumps(dict(all_passed=True,summaries=len(summary),view_records=len(records))))
