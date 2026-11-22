"""独立source-ID排序重放需求/prefix/静态pool，不读GT或quality。"""
from pathlib import Path
import json,hashlib,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'fr1_room'))
from fallback import generate
# 已知residual前缀、相同Raw/GAP tie按source ID、全ratio一致static pool。
ids=np.arange(100,200);mask=np.zeros(100,bool);mask[:20]=True;raw=np.linspace(1,3,100);gap=np.linspace(1,.1,100);score=raw*gap;score[:20]=np.linspace(5,.2,20);C=np.lexsort((ids,-score))
F,U,D,detail,pools,orders=generate(ids,mask,C,score,raw,gap)
assert np.array_equal(detail['RES'],D[.15])and len({len(v)for v in pools.values()})==1
for r in [.01,.05,.1,.15]:np.testing.assert_array_equal(orders['RF-C'][:int(len(ids)*r+.5)],C[:int(len(ids)*r+.5)])
tie_mask=np.arange(100)%2==0;tie=generate(ids,tie_mask,np.arange(100),np.ones(100),np.ones(100),np.ones(100));np.testing.assert_array_equal(tie[3]['RES'],np.flatnonzero(~tie_mask)[:len(tie[2][.15])]);assert len(tie[3]['RES'])>0
checks=[];frozen={}
for scene in ['fr1_room','fr1_xyz','fr2_xyz']:
    p=ROOT/scene;b=dict(np.load(p/'rankings.npz'));x=dict(np.load(p/'ranking_inputs.npz'));e=dict(np.load(p/'exact_residual.npz'));z=dict(np.load(p/'fallback_rankings.npz'));sets=dict(np.load(p/'selected_sets.npz'));ids=x['row_ids'];N=len(ids);F=set(map(int,ids[b['old_rep_mask']]));U=set(map(int,ids[~b['old_rep_mask']]));C=ids[b['C0_order']];mapping={int(v):i for i,v in enumerate(ids)}
    R=x['raw_I']*(1-e['cover']);np.testing.assert_array_equal(R[~b['old_rep_mask']],b['C0_score'][~b['old_rep_mask']]);np.testing.assert_array_equal(R,e['residual'])
    prefixes={a:sorted(U,key=lambda row:(-float(v[mapping[row]]),row))for a,v in [('RES',R),('RAW',x['raw_I']),('GAP',1-e['cover'])]}
    D={r:[int(v)for v in C[:int(r*N+.5)]if int(v)in U]for r in [.01,.05,.1,.15]};m=len(D[.15])
    for r,d in D.items():assert d==prefixes['RES'][:len(d)]
    for arm,name in [('RF-C','RES'),('RF-Raw','RAW'),('RF-Gap','GAP')]:
        pool=F|set(prefixes[name][:m]);expected=sorted(pool,key=lambda row:(-float(b['C0_score'][mapping[row]]),row));np.testing.assert_array_equal(ids[z[arm+'_pool']],sorted(pool));np.testing.assert_array_equal(ids[z[arm+'_order']],expected)
        for r in [.01,.05,.1,.15]:
            K=int(r*N+.5);np.testing.assert_array_equal(sets[f'{arm}_{r}'],sorted(expected[:K]));assert len(sets[f'{arm}_{r}'])==len(set(sets[f'{arm}_{r}']))==K
    np.testing.assert_array_equal(ids[z['D15_RES']],D[.15]);checks.append(dict(scene=scene,N=N,F=len(F),U=len(U),m15=m,product_row_exact=True,all_prefixes_exact=True,all_proxy_static_pools_exact=True,same_K=True))
    frozen[scene]={n:hashlib.sha256((p/n).read_bytes()).hexdigest()for n in ['selected_sets.npz','fallback_rankings.npz','exact_residual.npz']}
(ROOT/'independent_checks.json').write_text(json.dumps(dict(all_passed=True,checks=checks,pre_render_SHA=frozen,synthetic=['known_residual_prefix','stable_tie_source_ID','static_same_size_pools','RF_C_exact_prefix'],no_GT_quality_read=True),indent=2)+'\n')
print('INDEPENDENT_FALLBACK_PASS',checks,flush=True)
