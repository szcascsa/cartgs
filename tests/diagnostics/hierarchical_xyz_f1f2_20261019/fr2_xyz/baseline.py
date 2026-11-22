"""重建排序与重渲染旧四arm；gate通过前不生成新F1/F2。"""
from common import *
from fixed_marginal_reference import build
from progressive_ranker import progressive
from experiment_data import load
from evaluate import evaluate_arms
import csv,sys,struct
assert read(ROOT/'preparation.json')['all_passed']
os.environ['LD_LIBRARY_PATH']=str(ROOT/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
with open(ROOT/'synthetic_tests.log','w')as log:subprocess.run([sys.executable,'-m','unittest','test_progressive'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
x,base,nodes=load();ids=x['row_ids'];N=len(ids)
np.testing.assert_array_equal(np.lexsort((ids,-x['raw_I'])),base['A_Raw_order'])
t=tensors(ROOT/'contributions.pt');a=dict(np.load(ROOT/'frozen_input_arrays.npz'))
C=t['contribution'].numpy().astype(float);np.testing.assert_array_equal(C[:,ids].mean(0),x['raw_I']);np.testing.assert_array_equal(x['contribution'],C[:,ids])
np.testing.assert_array_equal(np.flatnonzero(a['valid']&a['observed']),ids)
for n in ['xyz','scale']:np.testing.assert_array_equal(t[n].numpy().astype(float)[ids],x[n])
b,_=build(x['xyz'],x['radius'],x['raw_I'],ids,read(ROOT/'population.json')['h'])
np.testing.assert_array_equal(b['order'],base['B_FixedMarginal_order']);np.testing.assert_array_equal(b['gain'],base['B_FixedMarginal_score'])
for cell in range(len(b['keys'])):
    v=b['rank_local'][b['order'][b['cell'][b['order']]==cell]];np.testing.assert_array_equal(v,np.arange(1,len(v)+1))
with open(ROOT/'C_rebuild.log','w')as log:subprocess.run([read(ROOT/'preparation.json')['baseline_C_executable'],str(ROOT/'hierarchy_input.bin'),str(ROOT/'structure_reproduced')],stdout=log,stderr=subprocess.STDOUT,check=True)
for name in ['sparse_hash_target_2048.bin','sparse_hash_target_2048_tree.csv']:assert (ROOT/'structure_reproduced'/name).read_bytes()==(ROOT/'structure_run1'/name).read_bytes(),name
with open(ROOT/'structure_reproduced/sparse_hash_target_2048.bin','rb')as f:
    nt=struct.unpack('<q',f.read(8))[0];score=np.frombuffer(f.read(nt*8),'<f8')
np.testing.assert_array_equal(np.lexsort((ids,-score[ids])),base['C_Hierarchical_order']);np.testing.assert_array_equal(score[ids],base['C_Hierarchical_score'])
f3=progressive(nodes,x['xyz'],x['radius'],x['raw_I'],ids,True);np.testing.assert_array_equal(f3['order'],base['F3_ProgressiveMarginal_order'])
np.testing.assert_array_equal(f3['gain'],np.load(ROOT/'prior_F3_rankings.npz')['F3_ProgressiveMarginal_priority'])
scene=Scene()
try:
    assert scene.call(dict(method='verify'))['maturity_all_one']
    maps=Path(f'/proc/{scene.proc.pid}/maps').read_text();(ROOT/'actual_worker_maps.txt').write_text(maps)
    for lib in ['libgaussian_mapper.so','libcuda_rasterizer.so']:assert str(ROOT/'runtime/lib'/lib)in maps
    metrics=Metrics(tensors(read(ROOT/'input_manifest.json')['cameras'])['gt'])
    records,summary=evaluate_arms(scene,metrics,{a:base[a+'_order']for a in read(ROOT/'preregistration.json')['baseline_reproduction_arms']},'baseline')
finally:scene.close()
old={(z['arm'],float(z['ratio']),int(z['frame_index'])):z for z in csv.DictReader(open(ROOT/'prior_quality_per_view.csv'))}
oldQ={(z['arm'],float(z['ratio'])):z for z in csv.DictReader(open(ROOT/'prior_quality_summary.csv'))}
diff={m:max(abs(r[m]-float(old[r['arm'],r['ratio'],r['frame_index']][m]))for r in records)for m in ['psnr','ssim','lpips','photo']}
agg={m:max(abs(r[m]-float(oldQ[r['arm'],r['ratio']][m]))for r in summary)for m in diff}
ok=all(v<=1e-6 for v in list(diff.values())+list(agg.values()))
save(ROOT/'baseline_reproduction.json',dict(all_passed=ok,complete_rankings_exact=True,C_tree_binary_exact=True,F3_prior_gain_exact=True,B_predecessors_exact=True,per_view_max_abs=diff,aggregate_max_abs=agg,rerenders=len(records),new_F1_F2_started=False))
if not ok:raise RuntimeError('STOP_BASELINE_REPRODUCTION_MISMATCH '+repr(diff))
print('BASELINE_EXACT_PASS',diff,flush=True)
