"""仅验证 native checkpoint 的固定十点布局和 Python 推理；不参与训练。"""
from pathlib import Path
import sys
import torch

repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo/'python'))
from selector import load_selector

directory=Path(sys.argv[1]);device=sys.argv[2]
model=load_selector(str(directory/'explicit_selector.pt'),device)
fixture=torch.jit.load(str(directory/'explicit_fixture.pt'),map_location=device).state_dict()
assert model.explicit_blocks and model.local_emd[0].in_features==27
for i,ratio in enumerate((.01,.05,.1,.15)):
    with torch.no_grad():
        p=model(fixture['xyz'],None,fixture['scale'],torch.tensor([[ratio]],device=device),
                sh_dc=fixture['dc'],opacity=fixture['opacity'])
    assert torch.allclose(p,fixture['p'+str(i)],rtol=1e-6,atol=1e-7)
print('PASS: explicit ten-slot native/Python parity, all four ratios')
