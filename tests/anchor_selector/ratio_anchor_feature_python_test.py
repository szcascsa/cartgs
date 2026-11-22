"""原生/Python 特征、插值、选后补偿和旧模型加载的一致性验证。"""
import sys
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from selector import load_selector

device=sys.argv[1] if len(sys.argv)>1 else 'cpu'
model=load_selector('ratio_selector.pt',device)
fixture=torch.jit.load('ratio_fixture.pt',map_location=device).state_dict()
assert model.local_emd[0].in_features==26
assert model.anchors.features.shape[1]==8
parameters=dict(model.anchors.named_parameters())
assert len(parameters)==5 and all(value.shape[1]==8 for value in parameters.values())
with torch.no_grad():
    for i,r in enumerate(fixture['ratios']):
        ratio=r.reshape(1,1).to(device)
        p,shared=model.forward_selection_features(fixture['xyz'],fixture['scale'],fixture['dc'],fixture['opacity'],ratio)
        assert torch.allclose(p,fixture['probability'][i],atol=1e-6,rtol=1e-5),float(r)
        assert torch.allclose(shared,fixture['shared'][i],atol=1e-6,rtol=1e-5),float(r)
        residual=model.compensation(shared[:3])
        assert torch.allclose(residual,fixture['residual'][i],atol=1e-6,rtol=1e-5)
    old=load_selector('ratio_legacy_selector.pt',device)
    assert old.local_emd[0].in_features==27 and old.anchors.features.shape[1]==16 and not old.ratio_features
torch.save(model.state_dict(),'ratio_python_roundtrip.pt')
restored=load_selector('ratio_python_roundtrip.pt',device)
assert all(torch.equal(value,restored.state_dict()[name]) for name,value in model.state_dict().items())
print('PASS: 9 ratios native/Python shared encoding, probability and selected compensation; independent tables; old 27D and new 26D checkpoint round trips')
