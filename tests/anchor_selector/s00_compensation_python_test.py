"""S00R00E08D12的原生/Python选后补偿一致性核验。"""
import sys
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from selector import load_selector

width=50 if len(sys.argv)>1 and sys.argv[1]=='e50' else 8
prefix='s00_e50' if width==50 else 's00'
model=load_selector(prefix+'_selector.pt','cpu')
fixture=torch.jit.load(prefix+'_fixture.pt',map_location='cpu').state_dict()
assert model.anchors.features.shape[1]==0 and not model.anchors.features.requires_grad
assert model.local_emd[0].in_features==11
assert model.compensation[0].in_features==32+width and len(model.compensation)==5 and model.compensation[4].out_features==14
assert int(model.anchor_layout_version)==10 and int(model.anchor_ratio_compensation_schema)==(3 if width==50 else 2)
for j in range(4):assert getattr(model.anchors,'compensation_features_'+str(j)).shape[1]==width
with torch.no_grad():
    for i,r in enumerate([.01,.025,.05,.075,.1,.125,.15]):
        ratio=torch.tensor([[r]])
        p,h=model.forward_selection_features(fixture['xyz'],fixture['scale'],fixture['dc'],fixture['alpha'],ratio)
        residual=model.compensation(torch.cat([h[fixture['ids']],model.anchors.compensation_lookup(fixture['ids'],float(ratio))],1))
        assert torch.allclose(p,fixture['probabilities'][i],atol=1e-6,rtol=1e-5)
        assert torch.allclose(h,fixture['shared'][i],atol=1e-6,rtol=1e-5)
        assert torch.allclose(residual,fixture['residuals'][i],atol=1e-6,rtol=1e-5)
    ids=fixture['ids'];owners=model.anchors.rows[ids]
    midpoint=(model.anchors.compensation_features_0[owners]+model.anchors.compensation_features_1[owners])*.5
    assert torch.allclose(model.anchors.compensation_lookup(ids,.03),midpoint)
    assert torch.equal(model.anchors.compensation_lookup(ids,.001),model.anchors.compensation_features_0[owners])
    assert torch.equal(model.anchors.compensation_lookup(ids,.9),model.anchors.compensation_features_3[owners])
print('PASS S00 native/Python: seven ratios, selected-row residuals, latent interpolation and endpoints')
legacy=load_selector('r3_reference_selector.pt','cpu')
assert legacy.local_emd[0].in_features==27 and len(legacy.compensation)==3
assert legacy.gaussian_compensation.independent and int(legacy.private_compensation_mode)==3
assert all(not p.requires_grad for p in legacy.compensation.parameters())
print('PASS R3 legacy checkpoint: Anchor16, frozen one-layer C, Gaussian/ratio Independent14D')
