from pathlib import Path
import csv,json
def write_csv(path,rows):
    with open(path,'w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def traces(root,arm,result,nodes,ids,roots,d50):
    ev=[];un=[]
    for a in result['events']:
        i=a['index'];n=nodes[a['node']];ev.append(dict(arm=arm,row_id=int(ids[i]),global_rank=a['rank'],priority=a['priority'],node=n.index,root=int(roots[i]),level=a['level'],phase=a['phase'],local_position=a['local_position']))
    for a in result['unlocks']:
        n=nodes[a['node']];side=d50*(32 if arm=='P32'else 0)/(2**(n.level-1))if arm=='P32'else None
        un.append(dict(arm=arm,node=n.index,level=n.level,side_length=side,side_over_d50=side/d50 if side else None,occupancy=len(n.members),Raw_SI_mass=n.mass,Q=n.quality,coarse_emitted_count=len(a['coarse_rows']),coarse_rows=json.dumps([int(ids[j])for j in a['coarse_rows']]),coarse_global_ranks=json.dumps(a['coarse_ranks']),coarse_Delta=json.dumps(a['coarse_gains']),activation_rank=a['activation_rank'],child_unlock_rank=a['unlock_rank'],number_of_children=len(a['children']),first_child_item_rank=a['first_child_item_rank']))
    write_csv(root/(arm+'_emit_trace.csv'),ev)
    if un:write_csv(root/(arm+'_unlock_trace.csv'),un)
