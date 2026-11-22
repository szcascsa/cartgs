"""只验证保存的 octree feature 映射与原 root offset 的推理一致性。"""
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"python"))
from selector import load_selector
folder, device = Path(sys.argv[1]), sys.argv[2]
model = load_selector(str(folder/"octree_selector_test.pt"),device)
probe = torch.jit.load(str(folder/"octree_selector_probe.pt"),map_location=device).state_dict()
assert model.anchors.levels.max() == 2
for i,r in enumerate((.01,.05,.1,.15)):
    with torch.no_grad():
        p=model(probe["xyz"],None,probe["scale"],torch.tensor([[r]],device=device),sh_dc=probe["sh_dc"],opacity=probe["opacity"])
        assert torch.allclose(p,probe["probability_"+str(i)],rtol=1e-6,atol=1e-7)
print("PASS: octree native/Python probabilities, saved leaf features and original root offsets")
