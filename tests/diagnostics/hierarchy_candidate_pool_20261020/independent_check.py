"""按source row独立重放pool union/sort/prefix；不读取图像或quality。"""
from pathlib import Path
import csv,json,sys,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'fr1_room'))
from pools import generate
# multi-hot / 同row跨level去重 / stable source row tie。
ids=np.array([10,20,30,40]);raw=np.array([3.,3.,2.,1.]);score=np.array([1.,2.,4.,3.])
records=[dict(node='0',level='1',leaf='0',rep1='10',rep2='20',rep3=''),dict(node='1',level='2',leaf='1',rep1='10',rep2='30',rep3=''),dict(node='2',level='2',leaf='1',rep1='40',rep2='',rep3='')]
F,M,l,pos,occ,orders=generate(records,ids,raw,score);np.testing.assert_array_equal(ids[F],[10,30,40]);np.testing.assert_array_equal(ids[M],[10,20,30,40]);assert l[0,0]and l[0,1]and len(occ[0])==2
np.testing.assert_array_equal(ids[orders['M-Raw']],[10,20,30,40]);np.testing.assert_array_equal(ids[orders['F-C']],[30,40,10]);assert len(F)<4
checks=[];frozen={}
for scene in ['fr1_room','fr1_xyz','fr2_xyz']:
    p=ROOT/scene;b=dict(np.load(p/'rankings.npz'));x=dict(np.load(p/'ranking_inputs.npz'));z=dict(np.load(p/'pool_rankings.npz'));sets=dict(np.load(p/'selected_sets.npz'));ids=x['row_ids'];N=len(ids)
    tree=list(csv.DictReader(open(p/'old_tree/hierarchy_tree.csv')));F=set();M=set();occ=[]
    for node in tree:
        for k in ['rep1','rep2','rep3']:
            if not node[k]:continue
            row=int(node[k]);M.add(row);occ.append(row)
            if node['leaf']=='1':F.add(row)
    assert F<=M and M<=set(map(int,ids));np.testing.assert_array_equal(ids[z['P_final']],sorted(F));np.testing.assert_array_equal(ids[z['P_multi']],sorted(M))
    source_to_index={int(row):i for i,row in enumerate(ids)}
    for label,pool in [('F',F),('M',M)]:
        for sc,values in [('Raw',x['raw_I']),('C',b['C0_score'])]:
            expected=np.array(sorted(pool,key=lambda row:(-float(values[source_to_index[row]]),row)))
            np.testing.assert_array_equal(ids[z[label+'-'+sc+'_order']],expected);previous=set()
            for ratio in [.01,.05,.1,.15]:
                K=int(ratio*N+.5);key=f'{label}-{sc}_{ratio}'
                if len(pool)<K:assert key not in sets;continue
                a=sets[key];np.testing.assert_array_equal(a,np.sort(expected[:K]));assert len(a)==len(set(a))==K and previous<=set(a);previous=set(a)
    checks.append(dict(scene=scene,F=len(F),M=len(M),node_rep_occurrences=len(occ),structural_union_exact=True,Raw_C_sort_exact=True,source_tie_exact=True,same_K_nested_exact=True,capacity_no_refill=True))
    frozen[scene]={name:hashlib.sha256((p/name).read_bytes()).hexdigest()for name in ['selected_sets.npz','pool_rankings.npz','candidate_pool_rows.csv']}
(ROOT/'independent_checks.json').write_text(json.dumps(dict(all_passed=True,checks=checks,synthetic=['cross_level_duplicate_union','multi_hot_identity','Raw_C_scoring','stable_source_tie','capacity_insufficient_no_refill'],no_GT_or_quality_read=True,pre_render_SHA=frozen),indent=2)+'\n')
print('INDEPENDENT_POOL_SORT_PREFIX_PASS',checks,flush=True)
