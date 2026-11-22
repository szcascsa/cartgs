"""32维单表特征的原生／Python推理一致性。"""
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from selector import load_selector
device=sys.argv[1] if len(sys.argv)>1 else 'cpu'
model=load_selector('feature32_selector.pt',device)
assert not model.ratio_features and model.anchors.features.shape[1]==32 and model.local_emd[0].in_features==43
assert len(list(model.anchors.parameters()))==1
fixture=torch.jit.load('feature32_fixture.pt',map_location=device).state_dict()
with torch.no_grad():
    for i,ratio in enumerate([.01,.05,.1,.15]):
        p,shared=model.forward_selection_features(fixture['xyz'],fixture['scale'],fixture['dc'],fixture['alpha'],torch.tensor([[ratio]],device=device))
        assert torch.allclose(p,fixture['probabilities'][i],atol=1e-6,rtol=1e-5)
        assert torch.allclose(shared,fixture['shared'][i],atol=1e-6,rtol=1e-5)
        assert torch.allclose(model.compensation(shared[:2]),fixture['residuals'][i],atol=1e-6,rtol=1e-5)
print('PASS: native/Python Feature32 selection, shared encoding, selected compensation')
