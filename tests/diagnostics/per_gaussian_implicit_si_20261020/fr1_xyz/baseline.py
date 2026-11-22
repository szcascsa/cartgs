"""A/B/C/F exact重建、same-K与逐view真实render复现。"""
from common import *
from fixed_marginal_reference import build
from evaluate import evaluate_arms
import csv
os.environ['LD_LIBRARY_PATH']=str(ROOT/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
x=dict(np.load(ROOT/'ranking_inputs.npz'));b=dict(np.load(ROOT/'rankings.npz'));old=dict(np.load(ROOT/'prior_pool_rankings.npz'));prior_sets=dict(np.load(ROOT/'prior_selected_sets.npz'));ids=x['row_ids'];N=len(ids)
np.testing.assert_array_equal(x['raw_I'],x['contribution'].mean(0));np.testing.assert_array_equal(np.lexsort((ids,-x['raw_I'])),b['A_order'])
arrays=dict(np.load(ROOT/'frozen_arrays.npz'));np.testing.assert_array_equal(np.flatnonzero(arrays['valid']&arrays['observed']),ids)
for name in ['xyz','scale']:np.testing.assert_array_equal(arrays[name][ids],x[name])
fixed,_=build(x['xyz'],x['radius'],x['raw_I'],ids,read(ROOT/'population.json')['h']);np.testing.assert_array_equal(fixed['order'],b['B_order']);np.testing.assert_array_equal(fixed['gain'],b['B_score'])
(ROOT/'reproduced_tree').mkdir()
with open(ROOT/'C_tree_rebuild.log','w')as log:subprocess.run(['/home/prmi/zwj/hierarchical-fr1-xyz-same-k-20261019/build/hierarchy_fixture',str(ROOT/'hierarchy_input.bin'),str(ROOT/'reproduced_tree')],stdout=log,stderr=subprocess.STDOUT,check=True)
for a,z in [('importance.bin','sparse_hash_target_2048.bin'),('hierarchy_tree.csv','sparse_hash_target_2048_tree.csv')]:assert(ROOT/'old_tree'/a).read_bytes()==(ROOT/'reproduced_tree'/z).read_bytes()
with open(ROOT/'old_tree/importance.bin','rb')as f:nt=int(np.fromfile(f,np.int64,1)[0]);score=np.fromfile(f,np.float64,nt)
np.testing.assert_array_equal(score[ids],b['C0_score']);np.testing.assert_array_equal(np.lexsort((ids,-score[ids])),b['C0_order'])
F=np.flatnonzero(b['old_rep_mask']);np.testing.assert_array_equal(F,old['P_final']);Fo=F[np.lexsort((ids[F],-b['C0_score'][F]))];np.testing.assert_array_equal(Fo,old['F-C_order'])
orders=dict(A=b['A_order'],B=b['B_order'],C=b['C0_order'],F=Fo)
for arm,o in orders.items():
    previous=set()
    for ratio in [.01,.05,.1,.15]:
        K=int(ratio*N+.5);rows=np.sort(ids[o[:K]]);np.testing.assert_array_equal(rows,prior_sets[f'{"F-C"if arm=="F"else arm}_{ratio}']);assert len(rows)==len(np.unique(rows))==K and previous<=set(rows);previous=set(rows)
scene=Scene()
try:
    assert scene.call(dict(method='verify'))['maturity_all_one'];maps=Path(f'/proc/{scene.proc.pid}/maps').read_text();(ROOT/'actual_worker_maps.txt').write_text(maps)
    for lib in ['libgaussian_mapper.so','libcuda_rasterizer.so']:assert str(ROOT/'runtime/lib'/lib)in maps
    metrics=Metrics(tensors(read(ROOT/'input_manifest.json')['cameras'])['gt']);records,summary=evaluate_arms(scene,metrics,orders,'baseline')
finally:scene.close()
(ROOT/'baseline_tensor_checks.json').write_bytes((ROOT/'frozen_tensor_checks.json').read_bytes())
old={(a['arm'],float(a['ratio']),int(a['frame_index'])):a for a in csv.DictReader(open(ROOT/'prior_quality_per_view.csv'))};oldq={(a['arm'],float(a['ratio'])):a for a in csv.DictReader(open(ROOT/'prior_quality_summary.csv'))};names=dict(A='A',B='B',C='C',F='F-C')
diff={m:max(abs(a[m]-float(old[names[a['arm']],a['ratio'],a['frame_index']][m]))for a in records)for m in ['psnr','ssim','lpips','photo']};agg={m:max(abs(a[m]-float(oldq[names[a['arm']],a['ratio']][m]))for a in summary)for m in diff};assert all(v==0 for v in list(diff.values())+list(agg.values())),('STOP_BASELINE',diff,agg)
save(ROOT/'baseline_reproduction.json',dict(all_passed=True,A_B_C_F_ranking_exact=True,selected_sets_exact=True,tree_binary_exact=True,per_view_max_abs=diff,aggregate_max_abs=agg,view_renders=len(records)))
print('BASELINE_EXACT_PASS',diff,flush=True)
