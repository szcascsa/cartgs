"""仅评价构建阶段已经固定的same-K pools；不足容量不补点。"""
from common import *
import hashlib
from evaluate import write_csv
import csv
assert read(ROOT.parent/'baseline_all_passed.json')['all_passed']and read(ROOT/'pool_checks.json')['all_passed']
frozen=read(ROOT.parent/'independent_checks.json')['pre_render_SHA'][ROOT.name]
for name,expected in frozen.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected
os.environ['LD_LIBRARY_PATH']=str(ROOT/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
sets=dict(np.load(ROOT/'selected_sets.npz'));info=read(ROOT/'input_manifest.json');frames=info['evaluation_frames'];scene=Scene();records=[];summary=[]
try:
    assert scene.call(dict(method='verify'))['maturity_all_one'];metrics=Metrics(tensors(info['cameras'])['gt'])
    for arm in ['F-Raw','F-C','M-Raw','M-C']:
        for ratio in read(ROOT/'protocol.json')['ratios']:
            key=f'{arm}_{ratio}'
            if key not in sets:print('CAPACITY INSUFFICIENT',arm,ratio,flush=True);continue
            rows=sets[key];K=len(rows);case=[];save(ROOT/'status.json',dict(stage='pool_render',arm=arm,ratio=ratio,completed=len(summary)))
            for start in range(0,len(frames),4):
                indices=frames[start:start+4];out=scene.call(dict(method='images',rows=rows.tolist(),ratio=ratio,frames=indices,apply_tf=True));values=metrics.all(torch.from_numpy(np.array(scene.mapping[:len(indices)])).cuda(),indices)
                for j,f in enumerate(indices):case.append(dict(arm=arm,ratio=ratio,K=K,frame_index=f,KF_id=info['camera_ids'][f],photo=out['photo'][j],psnr=float(values[j,0]),ssim=float(values[j,1]),lpips=float(values[j,2])))
            records.extend(case);s=dict(arm=arm,ratio=ratio,K=K,views=len(case),**{m:float(np.mean([r[m]for r in case]))for m in ['photo','psnr','ssim','lpips']});summary.append(s)
            write_csv('new_quality_per_view.csv',records);write_csv('new_quality_summary.csv',summary);print('QUALITY',json.dumps(s),flush=True)
finally:scene.close()
for name,expected in frozen.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected
write_csv('quality_per_view.csv',list(csv.DictReader(open(ROOT/'baseline_per_view.csv')))+records)
write_csv('quality_summary.csv',list(csv.DictReader(open(ROOT/'baseline_quality.csv')))+summary)
save(ROOT/'quality_complete.json',dict(all_passed=True,baseline_renders=16*len(frames),new_pool_renders=len(records),capacity_skips=[z for z in read(ROOT/'pool_checks.json')['capacity']if z['status']!='SUFFICIENT'],optimizer_updates=0,no_new_Progressive=True))
