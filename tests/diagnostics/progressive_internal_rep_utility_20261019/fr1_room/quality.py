"""对显式same-K集合使用原renderer/TF/metric；L/G不假设nested。"""
from common import *
from evaluate import write_csv
import csv
assert read(ROOT.parent/'baseline_all_passed.json')['all_passed']and read(ROOT/'construction_checks.json')['all_passed']
os.environ['LD_LIBRARY_PATH']=str(ROOT/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
sets=np.load(ROOT/'diagnostic_selected_sets.npz');info=read(ROOT/'input_manifest.json');frames=info['evaluation_frames'];scene=Scene();records=[];summary=[]
try:
    assert scene.call(dict(method='verify'))['maturity_all_one'];metrics=Metrics(tensors(info['cameras'])['gt'])
    for arm in ['S','L','G']+['L_Q'+str(q)for q in range(4)]:
        for ratio in read(ROOT/'protocol.json')['ratios']:
            rows=sets[f'{arm}_{ratio}'];K=len(rows);case=[]
            save(ROOT/'status.json',dict(stage='diagnostic_render',arm=arm,ratio=ratio,completed=len(summary)))
            for start in range(0,len(frames),4):
                indices=frames[start:start+4];out=scene.call(dict(method='images',rows=rows.tolist(),ratio=ratio,frames=indices,apply_tf=True))
                values=metrics.all(torch.from_numpy(np.array(scene.mapping[:len(indices)])).cuda(),indices)
                for j,f in enumerate(indices):case.append(dict(arm=arm,ratio=ratio,K=K,frame_index=f,KF_id=info['camera_ids'][f],photo=out['photo'][j],psnr=float(values[j,0]),ssim=float(values[j,1]),lpips=float(values[j,2])))
            records.extend(case);s=dict(arm=arm,ratio=ratio,K=K,views=len(case),**{m:float(np.mean([r[m]for r in case]))for m in ['photo','psnr','ssim','lpips']});summary.append(s)
            write_csv('new_quality_per_view.csv',records);write_csv('new_quality_summary.csv',summary);print('QUALITY',json.dumps(s),flush=True)
finally:scene.close()
write_csv('quality_per_view.csv',list(csv.DictReader(open(ROOT/'baseline_per_view.csv')))+records)
write_csv('quality_summary.csv',list(csv.DictReader(open(ROOT/'baseline_quality.csv')))+summary)
save(ROOT/'quality_complete.json',dict(all_passed=True,baseline_renders=16*len(frames),new_diagnostic_renders=len(records),main_arms=['A','B','C','P0','S','L','G'],secondary_Q_probes=['L_Q'+str(q)for q in range(4)]))
