"""对照PyTorch官方Gumbel前向/反向、原checkpoint及C++导出的真实MLP结果。"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'python'))
from selector import load_selector, build_selector_mask
from selector_sampling import gumbel_selection, gumbel_ratio_loss, flexgs_inference_mask
from elastic_budget import DEFAULTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--baseline', required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    output = Path(args.evidence)
    checks = []

    def equal(name, actual, expected, exact=False):
        assert actual.shape == expected.shape, name
        ok = torch.equal(actual, expected) if exact else torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)
        assert ok, (name, (actual-expected).abs().max().item())
        checks.append(dict(name=name, max_abs=float((actual.double()-expected.double()).abs().max()) if actual.numel() else 0))

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    # Same RNG state: actual F.gumbel_softmax, including its ST backward.
    for temperature in (.3, 1., 2.):
        z = torch.linspace(-3, 3, 514, device=device).reshape(-1,2).requires_grad_()
        vjp = torch.linspace(-1, 1, 257, device=device)
        torch.manual_seed(20261028)
        expected = F.gumbel_softmax(z, tau=temperature, hard=True)[:,1]
        eg = torch.autograd.grad((expected*vjp).sum() + .01*F.l1_loss(expected.mean(), z.new_tensor(.05)), z)[0]
        torch.manual_seed(20261028)
        actual = gumbel_selection(z, temperature)
        ag = torch.autograd.grad((actual*vjp).sum() + .01*gumbel_ratio_loss(actual, .05), z)[0]
        equal(f'pytorch_hard/{temperature}', actual, expected, True)
        equal(f'pytorch_photo_budget_gradient/{temperature}', ag, eg, True)
    z = torch.zeros(4096,2,device=device)
    assert not torch.equal(gumbel_selection(z), gumbel_selection(z))
    checks.append(dict(name='fresh_noise_each_forward', passed=True))
    for n in (0,1,17,10000):
        z = torch.randn(n,2,device=device)
        noise = -torch.empty_like(z).exponential_().log()
        for ratio in (.01,.025,.05,.075,.1,.125,.15,1.):
            mask,scores,threshold = flexgs_inference_mask(z,ratio,noise=noise)
            if n:
                expected = ((z+noise).softmax(-1)[:,1] >= scores.sort().values[int((1-ratio)*(n-1))])
                equal(f'official_percentile/{n}/{ratio}',mask.float(),expected.float(),True)
            else:
                assert not mask.numel() and threshold is None
    mask,_,_ = flexgs_inference_mask(torch.zeros(17,2,device=device),.01,noise=torch.zeros(17,2,device=device))
    assert mask.all()  # Official >= retains ties; this is deliberately not strict Top-K.
    for allowed in (0,1,3):
        eligible = torch.arange(17,device=device)<allowed
        mask,_,_ = flexgs_inference_mask(torch.zeros(17,2,device=device),.5,eligible=eligible)
        assert not mask[~eligible].any() and int(mask.sum()) <= allowed
    checks.append(dict(name='empty_ties_and_inactive_candidates',passed=True))

    fixture = dict(torch.jit.load(str(output/'fixture.pt'),map_location=device).named_buffers())
    model = load_selector(str(output/'selector.pt'),device)
    x,scale,dc,opacity = [fixture[k].detach().clone().requires_grad_() for k in ('xyz','scale','dc','opacity')]
    ratio = x.new_tensor([[.05]])
    probability,shared,logits = model.forward_selection_features(x,scale,dc,opacity,ratio,return_logits=True)
    equal('native_logits', logits, fixture['logits'])
    equal('native_probability', probability, fixture['probability'])
    mask = gumbel_selection(logits,noise=fixture['noise'])
    equal('native_mask',mask,fixture['mask'],True)
    equal('native_hard_budget',gumbel_ratio_loss(mask,.05),fixture['budget'])
    normalized_leaf = fixture['logits'].detach().clone().requires_grad_()
    normalized_mask = gumbel_selection(normalized_leaf,noise=fixture['noise'])
    normalized_budget = gumbel_ratio_loss(normalized_mask,.05,normalize_target=True)
    equal('native_normalized_budget',normalized_budget,fixture['normalized_budget'])
    equal('native_normalized_gradient',torch.autograd.grad(normalized_budget,normalized_leaf)[0],
          fixture['normalized_gradient'])
    for count in (0,1,17,10000):
        for ratio_value in (.01,.05,.10,.15):
            raw = torch.zeros(count,2,device=device,requires_grad=True)
            noise = -torch.empty_like(raw).exponential_().log()
            selected = gumbel_selection(raw,noise=noise)
            loss = gumbel_ratio_loss(selected,ratio_value,normalize_target=True)
            target = ratio_value*count
            reference = (selected.sum()-target).abs()/max(1.,target)
            equal(f'normalized_formula/{count}/{ratio_value}',loss,reference)
            gradient = torch.autograd.grad(loss,raw,retain_graph=True)[0]
            equal(f'normalized_gradient/{count}/{ratio_value}',gradient,torch.autograd.grad(reference,raw)[0])
            assert torch.isfinite(gradient).all()
    grad = torch.autograd.grad((mask*fixture['vjp']).sum()+.01*gumbel_ratio_loss(mask,.05),[x,scale,dc,opacity])
    for name,g in zip(('xyz','scale','dc','opacity'),grad):
        equal('native_joint_gradient/'+name,g,fixture[name+'_gradient'])
    leaf = fixture['logits'].detach().clone().requires_grad_()
    g = torch.autograd.grad((gumbel_selection(leaf,noise=fixture['noise'])*fixture['vjp']).sum(),leaf)[0]
    equal('native_logits_gradient',g,fixture['gradient'])

    spec = importlib.util.spec_from_file_location('baseline_selector',args.baseline)
    baseline = importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
    original = baseline.load_selector(str(output/'selector.pt'),device)
    assert list(model.state_dict()) == list(original.state_dict())
    for key,value in model.state_dict().items():equal('state_dict/'+key,value,original.state_dict()[key],True)
    old = original.forward_selection_features(x,scale,dc,opacity,ratio)
    now = model.forward_selection_features(x,scale,dc,opacity,ratio)
    equal('baseline_probability',now[0],old[0],True)
    equal('baseline_hidden',now[1],old[1],True)
    oldg = torch.autograd.grad((old[0]*fixture['vjp']).sum(),[x,scale,dc,opacity])
    newg = torch.autograd.grad((now[0]*fixture['vjp']).sum(),[x,scale,dc,opacity])
    for i,(a,b) in enumerate(zip(newg,oldg)):equal('baseline_input_gradient/'+str(i),a,b,True)

    n = len(x)
    gs = SimpleNamespace(get_xyz=x,get_scaling=scale,get_features=dc,get_opacity=opacity)
    metadata = dict(maturity=torch.ones(n,device=device),drift_ema=torch.zeros(n,device=device),
                    cached_uniform=torch.linspace(.01,.99,n,device=device),mapping_iteration=6000)
    cfg = dict(DEFAULTS,entry_maturity=0.)
    checkpoints = {k:v.clone() for k,v in model.state_dict().items()}
    a,_=build_selector_mask(model,gs,metadata,.05,cfg,protection_enabled=False)
    b,_=baseline.build_selector_mask(original,gs,metadata,.05,cfg,protection_enabled=False)
    equal('legacy_default_mask',a,b,True)
    cfg['selection_mode']='flexgs'
    torch.manual_seed(97)
    a,stats=build_selector_mask(model,gs,metadata,.05,cfg,protection_enabled=False)
    changed=copy.copy(metadata);changed['cached_uniform']=1-metadata['cached_uniform']
    torch.manual_seed(97)
    b,_=build_selector_mask(model,gs,changed,.05,cfg,protection_enabled=False,
                             forward_features=model.forward_selection_features(x,scale,dc,opacity,ratio,return_logits=True))
    equal('inference_ignores_cached_uniform',a,b,True)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in checkpoints.items())
    inactive_model=copy.deepcopy(model)
    inactive_model.anchors.register_buffer('slot_inactive',torch.ones(n,device=device,dtype=torch.bool))
    inactive_model.anchors.register_buffer('slot_recovered',torch.zeros(n,device=device,dtype=torch.bool))
    cold=copy.copy(gs);cold.get_opacity=torch.full_like(opacity,1e-5)
    inactive,info=build_selector_mask(inactive_model,cold,metadata,.05,cfg,protection_enabled=False)
    assert not inactive.any() and info['selection_threshold'] is None
    recovered,_=build_selector_mask(inactive_model,gs,metadata,.05,cfg,protection_enabled=False)
    assert recovered.any() and not inactive_model.anchors.slot_inactive.any()
    checks.append(dict(name='integrated_slot_deactivation_and_recovery',passed=True))
    result=dict(device=args.device,checks=len(checks),passed=True,actual_k=int(a.sum()),population=n,
                sampling='fresh Hard Gumbel-Softmax forward/backward',budget='abs(mean(ST mask)-ratio)',
                details=checks)
    (output/'python_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='details'}),flush=True)


if __name__ == '__main__':
    main()
