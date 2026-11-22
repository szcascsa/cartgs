"""显式保存冻结node candidate rows和child relation，便于审计unlock。"""
from pathlib import Path
import json
import numpy as np
from experiment_data import load
ROOT=Path(__file__).resolve().parent
x,b,nodes=load();ids=x['row_ids']
np.savez_compressed(ROOT/'frozen_tree_membership.npz',**{'node_'+str(i):ids[n.members]for i,n in nodes.items()})
result=[{'node':i,'stable_key':n.key,'level':n.level,'children':n.children,'final_leaf':n.leaf,'Raw_SI_mass':n.mass,'Q':n.quality,
         'original_first3_rows':[int(ids[j])for j in n.original_reps],'original_first3_Delta':n.original_gains,'occupancy':len(n.members)}for i,n in nodes.items()]
(ROOT/'frozen_tree_nodes.json').write_text(json.dumps(result,indent=2)+'\n');print('FROZEN_TREE_EXPORT_PASS',len(nodes))
