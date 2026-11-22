"""PG-ST保存结果的真实same-K重渲染逐view exact复现。"""
from pathlib import Path
import os,sys,csv,json
import numpy as np,torch
ROOT=Path(__file__).resolve().parent;name=sys.argv[1];HERE=ROOT/name
sys.path.insert(0,str(HERE));from common import Scene,Metrics,tensors
torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
os.environ['LD_LIBRARY_PATH']=str(HERE/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
info=json.loads((HERE/'input_manifest.json').read_text());reg=json.loads((ROOT/'preregistration.json').read_text());sets=np.load(ROOT/'legacy_selected_sets.npz')
lookup={(z['split'],float(z['ratio']),int(z['seed']),int(z['frame_index'])):z for z in csv.DictReader(open(ROOT/'legacy_quality_per_view.csv'))if z['scene']==name and z['arm'].startswith('PG-')and int(z['step'])==2000}
records=[];maximum={m:0.0 for m in ['psnr','ssim','lpips','photo']};scene=Scene()
try:
    metrics=Metrics(tensors(info['cameras'])['gt'])
    for ratio in reg['ratios']:
        for seed in reg['seeds']:
            rows=sets[f'{name}_PG_{int(ratio*100)}_seed_{seed}_step_2000']
            for split in ['train','eval']:
                frames=info['bank_frames'if split=='train'else'evaluation_frames']
                for start in range(0,len(frames),4):
                    indices=frames[start:start+4];out=scene.call(dict(method='images',rows=rows.tolist(),ratio=ratio,frames=indices,apply_tf=True));v=metrics.all(torch.from_numpy(np.array(scene.mapping[:len(indices)])).cuda(),indices)
                    for j,f in enumerate(indices):
                        z=dict(scene=name,split=split,ratio=ratio,seed=seed,frame_index=f,psnr=float(v[j,0]),ssim=float(v[j,1]),lpips=float(v[j,2]),photo=out['photo'][j]);ref=lookup[split,ratio,seed,f]
                        for m in maximum:maximum[m]=max(maximum[m],abs(z[m]-float(ref[m])))
                        records.append(z)
finally:scene.close()
assert all(v==0 for v in maximum.values()),('STOP_LEGACY_NOT_EXACT',maximum)
with open(HERE/'legacy_reproduced_per_view.csv','w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
(HERE/'legacy_reproduction.json').write_text(json.dumps(dict(all_passed=True,per_view_max_abs=maximum,renders=len(records)),indent=2)+'\n');print('PG_ST_EXACT_PASS',name,flush=True)
