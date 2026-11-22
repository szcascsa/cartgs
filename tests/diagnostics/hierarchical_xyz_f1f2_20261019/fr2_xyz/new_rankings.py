"""基线gate通过后构建三条完整ranking；重复验证，保存emit和unlock全trace。"""
from pathlib import Path
import csv,json,time,hashlib
import numpy as np
from progressive_ranker import full_leaf_marginal,progressive,validate_ranking
from experiment_data import load
ROOT=Path(__file__).resolve().parent
def write(n,records):
    with open(ROOT/n,'w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
def save(n,v):(ROOT/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
assert json.loads((ROOT/'baseline_reproduction.json').read_text())['all_passed']
x,base,nodes=load();ids=x['row_ids'];w=x['raw_I'];N=len(ids)
treefile=ROOT/'structure_run1/sparse_hash_target_2048_tree.csv';tree_sha=hashlib.sha256(treefile.read_bytes()).hexdigest()
all_arrays={'row_ids':ids};metadata=[];tail=[];unlock=[];times={};checks={}
for arm,fn in [('F1_MarginalTail',lambda:full_leaf_marginal(nodes,x['xyz'],x['radius'],w,ids)),
               ('F2_ProgressiveResidual',lambda:progressive(nodes,x['xyz'],x['radius'],w,ids,False)),
               ('F3_ProgressiveMarginal',lambda:progressive(nodes,x['xyz'],x['radius'],w,ids,True))]:
    start=time.perf_counter();result=fn();times[arm]=(time.perf_counter()-start)*1000
    print('RANKING_BUILT',arm,times[arm],flush=True)
    repeat=fn();np.testing.assert_array_equal(result['order'],repeat['order']);np.testing.assert_array_equal(result['gain'],repeat['gain'])
    assert result['events']==repeat['events']and result['unlocks']==repeat['unlocks']
    
    if arm=='F3_ProgressiveMarginal':
        np.testing.assert_array_equal(result['order'],base[arm+'_order'])
        prior=np.load(ROOT/'prior_F3_rankings.npz');np.testing.assert_array_equal(result['gain'],prior[arm+'_priority'])
    validate_ranking(result['order'],N);assert hashlib.sha256(treefile.read_bytes()).hexdigest()==tree_sha
    order=result['order'];ordinal=np.zeros(N,dtype=np.int64);ordinal[order]=np.arange(N,0,-1)
    np.testing.assert_array_equal(np.lexsort((ids,-ordinal)),order)
    all_arrays[arm+'_order']=order;all_arrays[arm+'_score']=ordinal;all_arrays[arm+'_priority']=result['gain']
    for a in result['events']:
        i=a['index'];n=nodes[a['node']]
        metadata.append({'arm':arm,'row_id':int(ids[i]),'global_rank':a['rank'],'priority':a['priority'],'node':n.index,'root':int(base['root'][i]),'root_key':json.dumps(n.key[:3]),'path':n.key[3],'level':a['level'],'phase':a['phase'],'local_position':a['local_position'],'C_final_rep':bool(base['rep_mask'][i]),'C_rep_level':int(base['rep_level'][i])})
    for node,info in result['leaf_info'].items():
        n=nodes[node];gains=info['gains'];coarse=info['coarse_count'];tail_gain=float(sum(gains[coarse:]));d3=gains[2]if len(gains)>=3 else None;d4=gains[3]if len(gains)>=4 else None
        tail.append({'arm':arm,'leaf':node,'level':n.level,'occupancy':len(n.members),'leaf_Raw_SI_mass':n.mass,'Q':n.quality,'inherited_rows':info['inherited_rows'],'inherited_covered_mass':info['initial_ancestor_cover_mass'],'new_coarse_count':coarse,'emitted_rows':len(info['sequence']),
                     'original_Delta1':n.original_gains[0]if n.original_gains else None,'original_Delta2':n.original_gains[1]if len(n.original_gains)>1 else None,'original_Delta3':n.original_gains[2]if len(n.original_gains)>2 else None,
                     'Delta1':gains[0]if gains else None,'Delta2':gains[1]if len(gains)>1 else None,'Delta3':d3,'Delta4':d4,'Delta5':gains[4]if len(gains)>4 else None,'Delta4_over_Delta3':d4/d3 if d3 and d4 is not None else None,
                     'positive_gain_rows':int(sum(v>0 for v in gains)),'tail_total_gain':tail_gain,'tail_gain_over_Raw_mass':tail_gain/n.mass if n.mass else 0,'all_Delta':json.dumps(gains),'all_source_rows':json.dumps([int(ids[i])for i in info['sequence']])})
    for a in result['unlocks']:
        n=nodes[a['node']];item={'arm':arm,'node':n.index,'root_key':json.dumps(n.key[:3]),'path':n.key[3],'level':n.level,'Raw_SI_mass':n.mass,'Q':n.quality,'activation_rank':a['activation_rank'],'coarse_count':len(a['coarse_rows']),'child_unlock_rank':a['unlock_rank'],'number_of_children':len(a['children']),'first_child_item_rank':a['first_child_item_rank']}
        for j in range(3):
            item['coarse_rep'+str(j+1)]=int(ids[a['coarse_rows'][j]])if len(a['coarse_rows'])>j else None
            item['coarse_rank'+str(j+1)]=a['coarse_ranks'][j]if len(a['coarse_ranks'])>j else None
            item['coarse_Delta'+str(j+1)]=a['coarse_gains'][j]if len(a['coarse_gains'])>j else None
        unlock.append(item)
    checks[arm]={'complete_permutation':True,'unique_rows':N,'deterministic_repeat_exact':True,'nested_prefixes':True,'tree_SHA_unchanged':True,'ordinal_order_exact':True,'first3_C_exact':True if arm.startswith('F1')else 'ancestor-conditioned recomputation; frozen structure'}
    del result,repeat
np.savez_compressed(ROOT/'new_arm_rankings.npz',**all_arrays)
write('ranking_emit_trace.csv',metadata);write('marginal_tail_statistics.csv',tail);write('progressive_unlock_trace.csv',unlock)
save('new_ranking_checks.json',{'all_passed':True,'tree_SHA256':tree_sha,'nodes':len(nodes),'final_leaves':sum(n.leaf for n in nodes.values()),'N_candidate':N,'checks':checks,'runtime_ms':times,'production_modified':False})
print('NEW_RANKINGS_ALL_PASS',flush=True)
