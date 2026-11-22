"""读取native输出，验证R2/R3 checkpoint与完整选后属性路径。"""
import sys
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from selector import load_selector, compensated_attributes

device=sys.argv[1] if len(sys.argv)>1 else 'cpu'
for method in ['r3','r2']:
    name='private_'+method
    model=load_selector(name+'.pt',device)
    fixture=torch.jit.load(name+'_fixture.pt',map_location=device).state_dict()
    xyz,scale,dc,alpha=[fixture[k] for k in ['xyz','scale','dc','alpha']]
    ratio=torch.tensor([[.075]],device=device)
    probability,shared=model.forward_selection_features(xyz,scale,dc,alpha,ratio)
    ids=torch.arange(len(xyz),device=device)
    residual=model.compensation(shared)+model.gaussian_compensation(ids,.075)
    assert torch.allclose(probability,fixture['probability'],atol=2e-6,rtol=2e-6)
    assert torch.allclose(residual,fixture['expected'],atol=2e-6,rtol=2e-6)
    class Gaussian:
        get_xyz=xyz
        get_scaling=scale
        get_rotation=torch.tensor([[1.,0,0,0]],device=device).expand(len(xyz),4)
        get_features_dc=dc[:,None,:]
        _opacity=torch.logit(alpha)
    result=compensated_attributes(model,Gaussian(),torch.ones(len(xyz),device=device),.075,shared)
    assert torch.allclose(result['xyz'],xyz+fixture['expected'][:,:3],atol=2e-6,rtol=2e-6)
    if method=='r3':
        expected=(model.gaussian_compensation(ids,.05)+model.gaussian_compensation(ids,.1))*.5
        assert torch.allclose(model.gaussian_compensation(ids,.075),expected,atol=1e-7,rtol=0)
        assert torch.equal(model.gaussian_compensation(ids,1.),model.gaussian_compensation(ids,.15))
    else:
        assert torch.equal(model.gaussian_compensation(ids,.01),model.gaussian_compensation(ids,1.))
print('PASS native/Python R2/R3 probability,14D,attribute application,interpolation,endpoints')
