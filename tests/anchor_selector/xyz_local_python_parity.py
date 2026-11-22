"""30维模型的 native/Python 概率及 Cached Bernoulli mask 校验。"""
from pathlib import Path
import sys
import torch
repo=Path(__file__).resolve().parents[2];sys.path.insert(0,str(repo/"python"))
from selector import load_selector
folder,device=Path(sys.argv[1]),sys.argv[2]
model=load_selector(str(folder/"octree_selector_test.pt"),device)
probe=torch.jit.load(str(folder/"octree_selector_probe.pt"),map_location=device).state_dict()
assert model.global_xyz and model.local_emd[0].weight.shape[1]==30 and int(model.anchor_layout_version)==4
assert model.anchors.levels.max()==2
u=torch.linspace(.001,.999,len(probe["xyz"]),device=device)
for i,r in enumerate((.01,.05,.1,.15)):
    with torch.no_grad():
        p=model(probe["xyz"],None,probe["scale"],torch.tensor([[r]],device=device),sh_dc=probe["sh_dc"],opacity=probe["opacity"])
        error=float((p-probe["probability_"+str(i)]).abs().max())
        assert error<1e-6 and torch.equal(u<p,u<probe["probability_"+str(i)]),(r,error)
print("PASS: 30-input world XYZ, current cell center/scale, native/Python probability and Cached Bernoulli parity")
