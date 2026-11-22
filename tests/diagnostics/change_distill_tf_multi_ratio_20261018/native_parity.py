from pathlib import Path
import json,sys,torch
D=Path(__file__).resolve().parent;a,b=sys.argv[1:3];result={}
for branch in ['tf','both']:
 rows=[]
 for i in range(10):
  if not all((D/s/'raw'/f'native_{branch}'/f'warm_{i}'/'final_parameters.pt').exists() for s in [a,b]):continue
  x,y=[torch.jit.load(str(D/s/'raw'/f'native_{branch}'/f'warm_{i}'/'final_parameters.pt'),map_location='cpu').state_dict() for s in [a,b]]
  assert x.keys()==y.keys();bad=[];maximum=ratio=0
  for k,t in x.items():
   delta=(t-y[k]).abs();maximum=max(maximum,float(delta.max()))
   ratio=max(ratio,float((delta/(2e-5+2e-4*y[k].abs())).max()))
   if not torch.allclose(t,y[k],atol=2e-5,rtol=2e-4):bad.append(k)
  rows.append(dict(run=i,max_abs=maximum,max_gate_ratio=ratio,allclose=not bad,failed=bad))
 result[branch]=rows
(D/b/'raw'/('native_parity_'+a+'.json')).write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2));assert all(r['allclose'] for rows in result.values() for r in rows)
