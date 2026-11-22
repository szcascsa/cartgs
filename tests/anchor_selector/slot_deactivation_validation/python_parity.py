"""CPU/CUDA native checkpoint 的停用、恢复与 Cached Bernoulli 一致性。"""
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'python'))
from selector import load_selector,mask_inactive_slots
d=Path(sys.argv[1]);device=sys.argv[2]
model=load_selector(str(d/'slot_selector.pt'),device)
v=torch.jit.load(str(d/'slot_fixture.pt'),map_location=device).state_dict()
for j,r in enumerate([.01,.05,.1,.15]):
    with torch.no_grad():
        p=model(v['xyz'],None,v['scale'],torch.tensor([[r]],device=device),sh_dc=v['dc'],opacity=v['opacity'])
        p=mask_inactive_slots(model,p,v['opacity'])
    assert torch.allclose(p,v['p'+str(j)],atol=1e-7,rtol=1e-6)
    assert torch.equal(v['uniform']<p,v['mask'+str(j)])
blocked=model.anchors.slot_inactive.flatten().clone()
row=torch.nonzero(blocked).flatten()[0]
alpha=v['opacity'].clone();alpha[row]=1/255
p=mask_inactive_slots(model,torch.ones_like(v['uniform']),alpha)
assert p[row]==1 and not model.anchors.slot_inactive.flatten()[row]
alpha[row]=0
assert mask_inactive_slots(model,torch.ones_like(p),alpha)[row]==1
print('PASS slot native/Python masked probability, four ratios, cached uniforms, cutoff recovery')
