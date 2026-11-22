"""基线重复、候选重复、交叉比较；原 tolerance 不变。"""
from pathlib import Path
import json,sys,torch
D=Path(__file__).resolve().parent
baseline,candidate=sys.argv[1:3]
prefix='paired_'+candidate
def compare(a,b):
 x,y=[torch.jit.load(str(p),map_location='cpu').state_dict() for p in [a,b]]
 assert x.keys()==y.keys();rows=[]
 for k,t in x.items():
  d=(t-y[k]).abs()
  rows.append(dict(name=k,max_abs=float(d.max()) if d.numel() else 0,
    mean_abs=float(d.float().mean()) if d.numel() else 0,
    bitwise=torch.equal(t,y[k]),allclose=torch.allclose(t,y[k],atol=2e-5,rtol=2e-4)))
 return dict(allclose=all(r['allclose'] for r in rows),bitwise=all(r['bitwise'] for r in rows),
             max_abs=max(r['max_abs'] for r in rows),rows=rows)
result={}
for branch in ['tf','selector','both']:
 files={s:{d:D/s/'raw'/f'{prefix}_{d}_{branch}/warm_0/final_parameters.pt' for d in ['forward','reverse']} for s in [baseline,candidate]}
 for direction in ['forward','reverse']:
  result[branch+'_cross_'+direction]=compare(files[baseline][direction],files[candidate][direction])
 for stage in [baseline,candidate]:result[branch+'_repeat_'+stage]=compare(files[stage]['forward'],files[stage]['reverse'])
(D/('controls_'+candidate+'.json')).write_text(json.dumps(result,indent=2))
print({k:{a:b for a,b in v.items() if a!='rows'} for k,v in result.items()})
assert all(v['allclose'] for v in result.values())
