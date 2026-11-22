"""仅据冻结leaf路径独立重放S/L/G，不调用construct或读取质量/GT。"""
from pathlib import Path
import csv,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
SCENES=['fr1_room','fr1_xyz','fr2_xyz']
checks=[]
for scene in SCENES:
    p=ROOT/scene;b=dict(np.load(p/'rankings.npz'));ids=b['row_ids'];N=len(ids);sets=dict(np.load(p/'diagnostic_selected_sets.npz'));inv=list(csv.DictReader(open(p/'internal_rep_inventory.csv')));byid={int(a['source_row']):a for a in inv};tree=list(csv.DictReader(open(p/'old_tree/hierarchy_tree.csv')));P=ids[b['P0_order']];C=ids[b['C0_order']]
    noninternal=np.array([int(i)for i in P if byid[int(i)]['emit_type']!='INTERNAL_REP'])
    leafrec={int(a['node']):a for a in tree if a['leaf']=='1'}
    def descendants(a):
        return set(int(ids[j])for j in range(N)if int(leafrec[int(b['old_leaf'][j])]['root'])==int(a['root'])and leafrec[int(b['old_leaf'][j])]['path'].startswith(a['node_path']))
    for r in [.01,.05,.1,.15]:
        K=int(N*r+.5);ctop=set(map(int,C[:K]));selected=set(map(int,P[:K]));used=set();tr=[]
        np.testing.assert_array_equal(np.sort(noninternal[:K]),sets[f'S_{r}'])
        recorded=[z for z in csv.DictReader(open(p/'local_replacement_trace.csv'))if float(z['ratio'])==r]
        for i in map(int,P[:K]):
            a=byid[i]
            if a['emit_type']!='INTERNAL_REP'or i in ctop:continue
            D=descendants(a);candidate=next((int(j)for j in C if j in D and j not in selected and j not in used),None)
            z=recorded[len(tr)];assert int(z['original_internal_row'])==i and z['same_subtree']=='True'
            if candidate is None:assert z['success']=='False'and z['replacement_row']==''
            else:
                assert candidate==int(z['replacement_row'])and z['success']=='True';selected.remove(i);selected.add(candidate);used.add(candidate)
            tr.append(z)
        assert len(tr)==len(recorded);np.testing.assert_array_equal(np.array(sorted(selected)),sets[f'L_{r}'])
        initial={int(i)for i in P[:K]if not(byid[int(i)]['emit_type']=='INTERNAL_REP'and i not in ctop)}
        for i in C:
            if len(initial)==K:break
            initial.add(int(i))
        np.testing.assert_array_equal(np.array(sorted(initial)),sets[f'G_{r}'])
        for arm in ['A','B','C','P0','S','L','G']:
            a=sets[f'{arm}_{r}'];assert len(a)==len(np.unique(a))==K and np.isin(a,ids).all()
        checks.append(dict(scene=scene,ratio=r,K=K,S_stable_filter=True,L_independent_path_prefix_replay=True,L_trace_rows=len(tr),G_rank_refill=True,all_same_K=True))
(ROOT/'independent_checks.json').write_text(json.dumps(dict(all_passed=True,checks=checks,quality_or_GT_read=False),indent=2)+'\n')
print('INDEPENDENT_PATH_REPLAY_PASS',len(checks),flush=True)
