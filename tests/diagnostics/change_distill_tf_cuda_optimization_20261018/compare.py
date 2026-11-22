from pathlib import Path
import json
import sys
import torch
D=Path(__file__).resolve().parent
a,b=[D/x for x in sys.argv[1:3]]
branch=sys.argv[3] if len(sys.argv)>3 else 'tf'
result={}
for label,rel in [('verification',f'verify_{branch}/verification.pt'),('final',f'native_{branch}/warm_0/final_parameters.pt')]:
    x,y=[p/'raw'/rel for p in [a,b]]
    if not x.exists() or not y.exists(): continue
    x,y=[torch.jit.load(str(p),map_location='cpu').state_dict() for p in [x,y]]
    assert x.keys()==y.keys(),label
    rows=[]
    for k,t in x.items():
        d=(t-y[k]).abs()
        rows.append({'tensor':k,'max_abs':float(d.max()) if d.numel() else 0,'mean_abs':float(d.float().mean()) if d.numel() else 0,
                     'bitwise_equal':torch.equal(t,y[k]),'allclose':torch.allclose(t,y[k],atol=2e-5,rtol=2e-4)})
    result[label]={'allclose':all(r['allclose'] for r in rows),'bitwise_equal':all(r['bitwise_equal'] for r in rows),'max_abs':max(r['max_abs'] for r in rows),'tensors':rows}
out=b/'raw'/('parity_'+a.name+'_'+branch+'.json')
out.write_text(json.dumps(result,indent=2))
print(json.dumps({k:{a:b for a,b in v.items() if a!='tensors'} for k,v in result.items()},indent=2))
assert result and all(v['allclose'] for v in result.values())
