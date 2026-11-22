from common import *
os.environ['LD_LIBRARY_PATH']=str(ROOT/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH','')
out=ROOT/'coverage_export';out.mkdir()
with open(ROOT/'coverage_export.log','w')as log:subprocess.run([str(ROOT.parent/'build/coverage_fixture'),str(ROOT/'hierarchy_input.bin'),str(ROOT/'old_tree/importance.bin'),str(out)],stdout=log,stderr=subprocess.STDOUT,check=True)
import csv
with open(out/'coverage.bin','rb')as f:
    N=int(np.fromfile(f,np.int64,1)[0]);r,c,g,R=np.fromfile(f,np.float64,4*N).reshape(4,N)
x=dict(np.load(ROOT/'ranking_inputs.npz'));b=dict(np.load(ROOT/'rankings.npz'));ids=x['row_ids'];mask=~b['old_rep_mask']
np.testing.assert_array_equal(R[ids][mask],b['C0_score'][mask]);np.testing.assert_array_equal(g[ids],1-c[ids]);np.testing.assert_array_equal(R[ids],x['raw_I']*g[ids]);assert np.isfinite(c[ids]).all()and(c[ids]>=0).all()and(c[ids]<=1).all()
tree=list(csv.DictReader(open(ROOT/'old_tree/hierarchy_tree.csv')));rep=list(csv.DictReader(open(out/'leaf_reps.csv')))
for a in rep:
    leaf=tree[int(a['leaf'])];k=a['position'];assert int(a['source_row'])==int(leaf['rep'+k])and float(a['Delta'])==float(leaf['delta'+k])
np.savez_compressed(ROOT/'exact_residual.npz',radius=r[ids],cover=c[ids],gap=g[ids],residual=R[ids])
save(ROOT/'coverage_checks.json',dict(all_passed=True,original_selectCell_reused=True,all_final_scores_exact=True,final_rep_ID_Delta_exact=True,nonrep_residual_score_exact=True,NumPy_multiplication_matches=True,N_candidate=len(ids)))
print('COVER_RESIDUAL_ROW_EXACT_PASS',len(ids),flush=True)
