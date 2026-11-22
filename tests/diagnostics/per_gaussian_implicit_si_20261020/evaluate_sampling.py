"""Bank上初始/最终真实训练采样mask的诊断；不作为same-K主结果。"""
from pathlib import Path
import os,sys,json,csv
import numpy as np,torch
ROOT=Path(__file__).resolve().parent;name=sys.argv[1];HERE=ROOT/name
sys.path.insert(0,str(HERE));from common import Scene,Metrics,tensors
reg=json.loads((ROOT/'preregistration.json').read_text());info=json.loads((HERE/'input_manifest.json').read_text());sets=dict(np.load(ROOT/'training_sampling_sets.npz'))
os.environ['LD_LIBRARY_PATH']=str(HERE/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
def write(n,rows):
    with open(HERE/n,'w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
scene=Scene();summaries=[];records=[]
try:
    metrics=Metrics(tensors(info['cameras'])['gt'])
    for ratio in reg['ratios']:
        for seed in reg['seeds']:
            for step in [0,reg['steps']]:
                selected=sets[f'{name}_PG_{int(100*ratio)}_seed_{seed}_step_{step}'];case=[];frames=info['bank_frames']
                for start in range(0,len(frames),4):
                    indices=frames[start:start+4];out=scene.call(dict(method='images',rows=selected.tolist(),ratio=ratio,frames=indices,apply_tf=True));v=metrics.all(torch.from_numpy(np.array(scene.mapping[:len(indices)])).cuda(),indices)
                    for j,f in enumerate(indices):case.append(dict(scene=name,ratio=ratio,seed=seed,step=step,sampled_N=len(selected),frame_index=f,photo=out['photo'][j],psnr=float(v[j,0]),ssim=float(v[j,1]),lpips=float(v[j,2])))
                records.extend(case);summaries.append(dict(scene=name,ratio=ratio,seed=seed,step=step,sampled_N=len(selected),**{m:float(np.mean([z[m]for z in case]))for m in['photo','psnr','ssim','lpips']}))
                write('sampling_train_quality.csv',summaries);write('sampling_train_per_view.csv',records)
finally:scene.close()
