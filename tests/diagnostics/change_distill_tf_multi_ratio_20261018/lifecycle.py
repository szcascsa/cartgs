from pathlib import Path
import json
import subprocess
import sys
import torch
torch.set_num_threads(1)
D=Path(__file__).resolve().parent
candidate=sys.argv[1] if len(sys.argv)>1 else 'final'
E=Path('/home/prmi/zwj/mycartgs/results/change_distill_ablation/20261017-2038')
cases={'global':E/'controls/global_change','aabb4000':E/'events_default_v2/aabb_only/event_4000_1',
       'aabb5000':E/'events_default_v2/aabb_only/event_5000_2'}
results={}
for case,event in cases.items():
    for stage in ['baseline',candidate]:
        out=D/stage/'raw'/('lifecycle_'+case)
        if (out/'verification.pt').exists():continue
        assert not out.exists()
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        out.mkdir()
        command=[str(D/stage/'benchmark_verify'),str(event),str(out),'tf','5','8192','4','1','0','0']
        (out/'command.json').write_text(json.dumps(command,indent=2))
        with (out/'stdout.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    results[case]={}
    for relative in ['verification.pt','warm_0/final_parameters.pt']:
        a,b=[torch.jit.load(str(D/s/'raw'/('lifecycle_'+case)/relative),map_location='cpu').state_dict() for s in ['baseline',candidate]]
        assert a.keys()==b.keys()
        rows=[]
        for k,x in a.items():
            delta=(x-b[k]).abs()
            rows.append({'tensor':k,'max_abs':float(delta.max()) if delta.numel() else 0,'mean_abs':float(delta.float().mean()) if delta.numel() else 0,
                         'bitwise_equal':torch.equal(x,b[k]),'allclose':torch.allclose(x,b[k],atol=2e-5,rtol=2e-4)})
        results[case][relative]={'allclose':all(x['allclose'] for x in rows),'rows':rows}
        (D/('lifecycle_'+candidate+'.json')).write_text(json.dumps(results,indent=2))
        assert results[case][relative]['allclose'],case
    (D/('lifecycle_'+candidate+'.json')).write_text(json.dumps(results,indent=2))
    print('PASS',case,flush=True)
