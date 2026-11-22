from common import *
from baseline import previous_runtime
from deployment_runtime import DeploymentRenderer
from large_scene import LARGE


@torch.no_grad()
def main():
    group=sys.argv[1] if len(sys.argv)>1 else 'tum'
    s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
    a=previous_runtime(s);b=DeploymentRenderer(s.model,s.gaussians,s.config)
    cam=camera(s,len(s.views)//2);checks=0
    for view in [False,True]:
        for comp in [False,True]:
            with torch.inference_mode():
                x,ix,_=a.prepare(cam,budget=1000,view_conditioned=view,compensation=comp)
                y,iy,_=b.prepare(cam,budget=1000,view_conditioned=view,compensation=comp)
            assert torch.equal(ix,iy)
            for key,t in x['inputs'].items():
                if isinstance(t,torch.Tensor):assert torch.equal(t.view(torch.int32),y['inputs'][key].view(torch.int32))
            checks+=1
    rng=torch.cuda.get_rng_state().clone();state={n:t.clone() for n,t in s.model.state_dict().items()}
    for _ in range(6):b.prepare(cam,budget=2000)
    assert torch.equal(rng,torch.cuda.get_rng_state())
    assert all(torch.equal(v,state[n]) for n,v in s.model.state_dict().items())
    # Snapshot references used by C++ must remain independent of the source model.
    x,ix,_=b.prepare(cam,budget=2000)
    s.model.soft_net.weight.add_(.01);s.gaussians._xyz.add_(.03)
    y,iy,_=b.prepare(cam,budget=2000)
    assert torch.equal(ix,iy) and all(torch.equal(t,y['inputs'][k]) for k,t in x['inputs'].items() if isinstance(t,torch.Tensor))
    # Shared finite status must preserve the original exception and reset on the next request.
    old=b._selector[0][0,0].clone();b._selector[0][0,0]=float('nan')
    for enabled in [False,True]:
        for budget in [1000,100000]:
            try:b.prepare(cam,budget=budget,compensation=enabled)
            except RuntimeError as error:assert 'Nonfinite deployment Selector logits' in str(error)
            else:raise AssertionError('Selector finite check missing')
    b._selector[0][0,0].copy_(old)
    b.prepare(cam,budget=1000)
    noise=b._reference.noise.clone()
    for bits in [-1,2147483647]:
        b._reference.noise.view(torch.int32).fill_(bits)
        try:b.prepare(cam,budget=5000)
        except RuntimeError as error:assert 'Nonfinite deployment Selector logits' in str(error)
        else:raise AssertionError('Nonfinite score entered selection')
    b._reference.noise.copy_(noise)
    b.prepare(cam,budget=1000)
    old=b._base.tables[0].clone();b._base.tables[0].fill_(float('nan'))
    try:b.prepare(cam,budget=1000)
    except RuntimeError as error:assert 'Nonfinite deployment compensated attributes' in str(error)
    else:raise AssertionError('Compensation finite check missing')
    b._base.tables[0].copy_(old)
    b.prepare(cam,budget=1000)
    empty=camera(s,0);empty.full_proj_transform=torch.eye(4,device='cuda');empty.full_proj_transform[3,3]=-1e6
    _,ids,out=b.prepare(empty,budget=1000);assert len(ids)==0 and out['candidate_gaussians']==0
    try:DeploymentRenderer(s.model,s.gaussians,s.config,scheduler='python')
    except TypeError:pass
    else:raise AssertionError('Python scheduler still accepted')
    save(ROOT/'reports'/f'EXTRA_VALIDATION_{group}.json',dict(passed=True,inference_mode_cases=checks,snapshot_independent=True,
        rng_unchanged=True,source_state_unchanged=True,finite_checks=True,error_recovery=True,
        selector_nonfinite_trials=6,adversarial_nan_payload=True,empty_candidates=True,python_scheduler_removed=True))
    print('EXTRA_CHECKS_PASS',flush=True)


if __name__=='__main__':main()
