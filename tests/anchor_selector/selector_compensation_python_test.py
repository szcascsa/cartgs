"""复核原生 checkpoint、选后 K 行解码、非选中参数和 Python 导出属性。"""
import sys
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from selector import load_selector,compensated_attributes

device=sys.argv[1] if len(sys.argv)>1 else 'cpu'
model=load_selector('compensation_selector.pt',device)
fixture=torch.jit.load('compensation_fixture.pt',map_location=device).state_dict()
class Gaussians:
    get_xyz=fixture['xyz']
    get_scaling=fixture['scale']
    get_features=fixture['dc']
    get_features_dc=fixture['dc']
    get_opacity=fixture['alpha']
    _opacity=fixture['opacity_logits']
    get_rotation=fixture['rotation']
g=Gaussians()
with torch.no_grad():
    p,shared=model.forward_selection_features(g.get_xyz,g.get_scaling,g.get_features,g.get_opacity,
                                             torch.tensor([[.05]],device=device))
    assert torch.allclose(p,fixture['probability'],atol=1e-7,rtol=1e-6)
    assert torch.allclose(shared,fixture['shared'],atol=1e-7,rtol=1e-6)
    selected=torch.ones_like(p)
    result=compensated_attributes(model,g,selected,.05,shared)
    for name,expected in [('xyz','result_xyz'),('scaling','result_scale'),('rotation','result_rotation')]:
        assert torch.allclose(result[name],fixture[expected],atol=1e-6,rtol=1e-5),name
    appearance=model.compensation[-1].out_features==14
    if appearance:
        for name,expected in [('features_dc','result_dc'),('opacity_logits','result_opacity_logits')]:
            assert torch.allclose(result[name],fixture[expected],atol=1e-6,rtol=1e-5),name
    else:
        assert set(result)=={'xyz','scaling','rotation'}
    selected.zero_();selected[:2]=1
    calls=[]
    hook=model.compensation.register_forward_pre_hook(lambda _,args:calls.append(len(args[0])))
    result=compensated_attributes(model,g,selected,.05,shared)
    hook.remove()
    assert calls==[2]
    for name,original in [('xyz',g.get_xyz),('scaling',g.get_scaling),('rotation',g.get_rotation)]:
        assert torch.equal(result[name][2:],original[2:]),name
    if appearance:
        assert torch.equal(result['opacity_logits'][2:],g._opacity[2:])
        assert torch.equal(result['features_dc'][2:],g.get_features_dc[2:])
    from gaussian_model import GaussianModel
    export=GaussianModel(1);export.active_sh_degree=1
    export._xyz=g.get_xyz;export._scaling=g.get_scaling.log();export._rotation=g.get_rotation
    export._opacity=g._opacity;export._features_dc=g.get_features_dc
    export._features_rest=torch.arange(len(p)*9,device=device).float().reshape(len(p),3,3)/1000
    assert export.save_ply('compensation_export.ply',selected,result)==2
    reloaded=GaussianModel(1);reloaded.load_ply('compensation_export.ply',device)
    assert torch.equal(reloaded._opacity,(result['opacity_logits'] if appearance else g._opacity)[:2])
    assert torch.equal(reloaded.get_features_dc,(result['features_dc'] if appearance else g.get_features_dc)[:2])
    assert torch.equal(reloaded.get_features_rest,export._features_rest[:2])
    if device=='cuda':
        from renderer import prepare_render_data
        prepared=prepare_render_data(export,selected,result)['inputs']
        assert torch.equal(prepared['opacities'],reloaded.get_opacity)
        assert torch.equal(prepared['dc'],reloaded.get_features_dc)
        assert torch.equal(prepared['shs'],reloaded.get_features_rest)
    assert compensated_attributes(model,g,torch.zeros_like(p),.05,shared) is None
print('PASS: native/Python geometry and appearance, selected-only decoder, exact unselected/empty behavior, PLY logits/DC/high-order SH and prepared renderer parity')
