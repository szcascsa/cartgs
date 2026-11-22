"""先重建A/B/C并实际重渲染；不通过则停止所有新机制实验。"""
from common import *
from evaluate import evaluate_arms
import csv
P=Path('/home/prmi/zwj/hierarchical-si-quality-audit-20261019')
assert read(ROOT/'ranking_sanity.json')['all_passed']
old=np.load(P/'arm_rankings.npz');new=np.load(ROOT/'arm_rankings.npz')
for arm in ['A_Raw','B_FixedMarginal','C_Hierarchical']:
    np.testing.assert_array_equal(old[arm+'_order'],new[arm+'_order'])
    np.testing.assert_array_equal(old[arm+'_score'],new[arm+'_score'])
np.testing.assert_array_equal(old['row_ids'],new['row_ids'])
os.environ['LD_LIBRARY_PATH']=str(ROOT/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
scene=Scene()
try:
    assert scene.call({'method':'verify'})['maturity_all_one']
    maps=Path(f'/proc/{scene.proc.pid}/maps').read_text()
    for lib in ['libgaussian_mapper.so','libcuda_rasterizer.so']:assert str(ROOT/'runtime/lib'/lib)in maps
    (ROOT/'actual_worker_maps.txt').write_text(maps)
    metrics=Metrics(tensors(read(ROOT/'input_manifest.json')['cameras'])['gt'])
    records,summary=evaluate_arms(scene,metrics,{a:new[a+'_order']for a in ['A_Raw','B_FixedMarginal','C_Hierarchical']},'baseline')
finally:scene.close()
oldrows={(r['arm'],float(r['ratio']),int(r['frame_index'])):r for r in csv.DictReader(open(P/'quality_per_view.csv'))}
diff={m:max(abs(r[m]-float(oldrows[r['arm'],r['ratio'],r['frame_index']][m]))for r in records)for m in ['psnr','ssim','lpips','photo']}
oldsum={(r['arm'],float(r['ratio'])):r for r in csv.DictReader(open(P/'quality_summary.csv'))}
aggregate={m:max(abs(r[m]-float(oldsum[r['arm'],r['ratio']][m]))for r in summary)for m in diff}
ok=all(v<=1e-6 for v in list(diff.values())+list(aggregate.values()))
save(ROOT/'baseline_reproduction.json',dict(all_passed=ok,complete_ranking_exact=True,selected_sets_exact=True,per_view_max_abs=diff,aggregate_max_abs=aggregate,views=161,candidates=50330,TF_frozen=True))
if not ok:raise RuntimeError('STOP_BASELINE_MISMATCH '+repr(diff))
print('BASELINE_EXACT_PASS',diff,flush=True)
