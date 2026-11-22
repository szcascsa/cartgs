from common import *
from evaluate import write_csv
import csv,hashlib
assert read(ROOT.parent/'baseline_all_passed.json')['all_passed']and read(ROOT/'fallback_checks.json')['all_passed']
expected=read(ROOT.parent/'independent_checks.json')['pre_render_SHA'][ROOT.name]
for n,h in expected.items():assert sha(ROOT/n)==h
os.environ['LD_LIBRARY_PATH']=str(ROOT/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
sets=dict(np.load(ROOT/'selected_sets.npz'));info=read(ROOT/'input_manifest.json');frames=info['evaluation_frames'];scene=Scene();records=[];summary=[]
try:
    assert scene.call(dict(method='verify'))['maturity_all_one'];metrics=Metrics(tensors(info['cameras'])['gt'])
    for arm in ['RF-C','RF-Raw','RF-Gap']:
        for ratio in read(ROOT/'protocol.json')['ratios']:
            selected=sets[f'{arm}_{ratio}'];K=len(selected);case=[];save(ROOT/'status.json',dict(stage='fallback_render',arm=arm,ratio=ratio,completed=len(summary)))
            for start in range(0,len(frames),4):
                indices=frames[start:start+4];out=scene.call(dict(method='images',rows=selected.tolist(),ratio=ratio,frames=indices,apply_tf=True));values=metrics.all(torch.from_numpy(np.array(scene.mapping[:len(indices)])).cuda(),indices)
                for j,f in enumerate(indices):case.append(dict(arm=arm,ratio=ratio,K=K,frame_index=f,KF_id=info['camera_ids'][f],photo=out['photo'][j],psnr=float(values[j,0]),ssim=float(values[j,1]),lpips=float(values[j,2])))
            records.extend(case);s=dict(arm=arm,ratio=ratio,K=K,views=len(case),**{m:float(np.mean([a[m]for a in case]))for m in ['photo','psnr','ssim','lpips']});summary.append(s);write_csv('new_quality_per_view.csv',records);write_csv('new_quality_summary.csv',summary);print('QUALITY',json.dumps(s),flush=True)
finally:scene.close()
baseline=list(csv.DictReader(open(ROOT/'baseline_per_view.csv')));baselineq=list(csv.DictReader(open(ROOT/'baseline_quality.csv')));lookup={(float(a['ratio']),int(a['frame_index'])):a for a in baseline if a['arm']=='C'};qlookup={float(a['ratio']):a for a in baselineq if a['arm']=='C'}
diff={m:max(abs(a[m]-float(lookup[a['ratio'],a['frame_index']][m]))for a in records if a['arm']=='RF-C')for m in ['psnr','ssim','lpips','photo']};agg={m:max(abs(a[m]-float(qlookup[a['ratio']][m]))for a in summary if a['arm']=='RF-C')for m in diff}
assert all(v==0 for v in list(diff.values())+list(agg.values())),('STOP_RF_C_RENDER_NOT_EXACT',diff,agg)
for n,h in expected.items():assert sha(ROOT/n)==h
write_csv('quality_per_view.csv',baseline+records);write_csv('quality_summary.csv',baselineq+summary)
save(ROOT/'quality_complete.json',dict(all_passed=True,RF_C_real_render_per_view_max_abs=diff,RF_C_aggregate_max_abs=agg,baseline_renders=len(baseline),new_fallback_renders=len(records),training_updates=0))
