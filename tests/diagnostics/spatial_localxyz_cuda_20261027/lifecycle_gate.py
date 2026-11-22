"""同K不同ratio/身份、输入原地更新、append/prune/reorder替换的缓存核验。"""
import json
import argparse
import torch
from baseline_profile import ROOT,r,fixture
from implementation import Execution
from parity import compare


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',default='graph_all_fused');a=p.parse_args()
    sc,f,net,payload=fixture();ref=Execution(net);ex=Execution(net,a.method);rows=[]
    ids=f.sets[.01,'same_k'];originals=(f.features.own,f.features.coords,f.features.anchor_scale)
    sets={key:value.clone() for key,value in f.sets.items()}
    with torch.no_grad():
        saved=ex(.01,ids);snapshot=saved.clone()
        for j,(ratio,current) in enumerate([(.01,ids),(.075,ids),(.01,ids),(.05,ids.flip(0)),(.01,ids)]):
            compare(ref(ratio,current),ex(ratio,current),'switch_'+str(j),rows,a.method,len(current))
        assert torch.equal(saved,snapshot),'Graph output must not alias subsequent replay'
        assert torch.equal(ex.conditions[.01],ref.condition(.01))
        assert all(torch.equal(value,sets[key]) for key,value in f.sets.items())
        for label in ['inplace_update','append','prune','reorder']:
            values=[v.clone() for v in originals]
            if label=='inplace_update':
                values[0].add_(.001);values[1].add_(.001)
            if label=='append':values=[torch.cat([v,v[:3]],0) for v in values]
            if label=='prune':values=[v[:-3] for v in values]
            if label=='reorder':values=[v.flip(0).contiguous() for v in values]
            f.features.own,f.features.coords,f.features.anchor_scale=values
            compare(ref(.075,ids),ex(.075,ids),label,rows,a.method,len(ids))
        f.features.own,f.features.coords,f.features.anchor_scale=originals
        ex.invalidate();compare(ref(.075,ids),ex(.075,ids),'invalidate_rebuild',rows,a.method,len(ids))
    prefix='' if a.method=='graph_all_fused' else a.method+'_'
    r.rt.write_csv(ROOT/(prefix+'lifecycle_gate.csv'),rows)
    assert all(row['pass_'] for row in rows)
    (ROOT/(prefix+'lifecycle_gate.json')).write_text(json.dumps(dict(passed=True,tests=len(rows),
        scope='feature table and graph storage lifecycle only; not production Gaussian optimizer or online geometry integration'),indent=2))
    print('LIFECYCLE_GATE_PASS',len(rows),flush=True)


if __name__=='__main__':main()
