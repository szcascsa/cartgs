from pathlib import Path
import json,sys,torch
torch.set_num_threads(1)
D=Path(__file__).resolve().parent;a,b=sys.argv[1:3]
result={}
for branch in ['tf','both']:
 x,y=[torch.jit.load(str(D/s/'raw'/f'verify_{branch}/verification.pt'),map_location='cpu').state_dict() for s in [a,b]]
 groups={}
 for k,t in x.items():
  category=k.rsplit('_',1)[0]
  if k.startswith('tf_gradient_'):category='plane_gradient' if int(k.rsplit('_',1)[1])<12 else 'mlp_gradient'
  if k.startswith('feature_gradient_'):category='feature_gradient'
  g=groups.setdefault(category,dict(tensors=0,elements=0,max_abs=0,abs_sum=0,bitwise=True,allclose=True,max_gate_ratio=0))
  delta=(t-y[k]).abs();g['tensors']+=1;g['elements']+=t.numel()
  if t.numel():
   g['max_abs']=max(g['max_abs'],float(delta.max()));g['abs_sum']+=float(delta.double().sum())
   g['max_gate_ratio']=max(g['max_gate_ratio'],float((delta/(2e-5+2e-4*y[k].abs())).max()))
  g['bitwise'] &= torch.equal(t,y[k]);g['allclose'] &= torch.allclose(t,y[k],atol=2e-5,rtol=2e-4)
 for g in groups.values():g['mean_abs']=g.pop('abs_sum')/max(g['elements'],1)
 result[branch]=groups
(D/b/'raw'/('parity_categories_'+a+'.json')).write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
