"""三个scene诊断质量前的A/B/C/P0完整重建与逐view复现。"""
from common import *
from fixed_marginal_reference import build
from progressive_ranker import progressive,validate_ranking
from tree_io import nodes_from_tree
from evaluate import evaluate_arms
import csv
PARENT=ROOT.parent
os.environ['LD_LIBRARY_PATH']=str(ROOT/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
x=dict(np.load(ROOT/'ranking_inputs.npz'));b=dict(np.load(ROOT/'rankings.npz'));ids=x['row_ids'];N=len(ids)
np.testing.assert_array_equal(x['raw_I'],x['contribution'].mean(0));np.testing.assert_array_equal(np.lexsort((ids,-x['raw_I'])),b['A_order'])
a=dict(np.load(ROOT/'frozen_arrays.npz'));np.testing.assert_array_equal(np.flatnonzero(a['valid']&a['observed']),ids)
for n in ['xyz','scale']:np.testing.assert_array_equal(a[n][ids],x[n])
fixed,_=build(x['xyz'],x['radius'],x['raw_I'],ids,read(ROOT/'population.json')['h']);np.testing.assert_array_equal(fixed['order'],b['B_order']);np.testing.assert_array_equal(fixed['gain'],b['B_score'])
for cell in range(len(fixed['keys'])):
    v=fixed['rank_local'][fixed['order'][fixed['cell'][fixed['order']]==cell]];np.testing.assert_array_equal(v,np.arange(1,len(v)+1))
(ROOT/'reproduced_old').mkdir(exist_ok=True)
with open(ROOT/'old_C_rebuild.log','w')as log:subprocess.run(['/home/prmi/zwj/hierarchical-fr1-xyz-same-k-20261019/build/hierarchy_fixture',str(ROOT/'hierarchy_input.bin'),str(ROOT/'reproduced_old')],stdout=log,stderr=subprocess.STDOUT,check=True)
for a,z in [('importance.bin','sparse_hash_target_2048.bin'),('hierarchy_tree.csv','sparse_hash_target_2048_tree.csv')]:assert (ROOT/'old_tree'/a).read_bytes()==(ROOT/'reproduced_old'/z).read_bytes(),a
np.testing.assert_array_equal(np.lexsort((ids,-b['C0_score'])),b['C0_order'])
with open(ROOT/'old_tree/importance.bin','rb')as f:
    nt=int(np.fromfile(f,np.int64,1)[0]);score=np.fromfile(f,np.float64,nt)
np.testing.assert_array_equal(score[ids],b['C0_score'])
nodes,_=nodes_from_tree(ROOT/'old_tree/hierarchy_tree.csv',x['xyz'],ids,b['old_root'])
r=progressive(nodes,x['xyz'],x['radius'],x['raw_I'],ids,False);np.testing.assert_array_equal(r['order'],b['P0_order']);np.testing.assert_array_equal(r['gain'],b['P0_priority'])
save(ROOT/'P0_rebuilt_events.json',dict(events=r['events'],unlocks=r['unlocks']))
orders={arm:b[('C0'if arm=='C'else arm)+'_order']for arm in ['A','B','C','P0']}
for order in orders.values():validate_ranking(order,N)
scene=Scene()
try:
    assert scene.call(dict(method='verify'))['maturity_all_one']
    maps=Path(f'/proc/{scene.proc.pid}/maps').read_text();(ROOT/'actual_worker_maps.txt').write_text(maps)
    for n in ['libgaussian_mapper.so','libcuda_rasterizer.so']:assert str(ROOT/'runtime/lib'/n)in maps
    metrics=Metrics(tensors(read(ROOT/'input_manifest.json')['cameras'])['gt'])
    records,summary=evaluate_arms(scene,metrics,orders,'baseline')
finally:scene.close()
(ROOT/'baseline_tensor_checks.json').write_bytes((ROOT/'frozen_tensor_checks.json').read_bytes())
names=dict(A='A_Raw',B='B_FixedMarginal',C='C_Hierarchical',P0='F2_ProgressiveResidual')
old={(a['arm'],float(a['ratio']),int(a['frame_index'])):a for a in csv.DictReader(open(ROOT/'prior_quality_per_view.csv'))};oldq={(a['arm'],float(a['ratio'])):a for a in csv.DictReader(open(ROOT/'prior_quality_summary.csv'))}
diff={m:max(abs(a[m]-float(old[names[a['arm']],a['ratio'],a['frame_index']][m]))for a in records)for m in ['psnr','ssim','lpips','photo']};agg={m:max(abs(a[m]-float(oldq[names[a['arm']],a['ratio']][m]))for a in summary)for m in diff}
ok=all(z==0 for z in list(diff.values())+list(agg.values()))
save(ROOT/'baseline_reproduction.json',dict(all_passed=ok,complete_rankings_exact=True,same_K=True,selected_sets_exact=True,C_binary_tree_exact=True,P0_priority_exact=True,per_view_max_abs=diff,aggregate_max_abs=agg,view_renders=len(records)))
if not ok:raise RuntimeError('STOP_BASELINE_REPRODUCTION_MISMATCH '+repr(diff))
print('BASELINE_EXACT_PASS',diff,flush=True)
