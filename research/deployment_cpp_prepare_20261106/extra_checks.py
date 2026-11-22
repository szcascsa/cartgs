from common import *
from baseline import previous_runtime
from deployment_runtime import DeploymentRenderer


@torch.no_grad()
def main():
    s=scene();a=previous_runtime(s);b=DeploymentRenderer(s.model,s.gaussians,s.config,scheduler='cpp')
    cam=camera(s,396);checks=0
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
    # The native and Python paths must reject the same nonfinite per-frame state.
    old=b._selector[0][0,0].clone();b._selector[0][0,0]=float('nan')
    try:b.prepare(cam,budget=1000)
    except RuntimeError as error:assert 'Nonfinite' in str(error)
    else:raise AssertionError('Selector finite check missing')
    b._selector[0][0,0].copy_(old)
    old=b._base.tables[0].clone();b._base.tables[0].fill_(float('nan'))
    try:b.prepare(cam,budget=1000)
    except RuntimeError as error:assert 'Nonfinite' in str(error)
    else:raise AssertionError('Compensation finite check missing')
    b._base.tables[0].copy_(old)
    empty=camera(s,0);empty.full_proj_transform=torch.eye(4,device='cuda');empty.full_proj_transform[3,3]=-1e6
    _,ids,out=b.prepare(empty,budget=1000);assert len(ids)==0 and out['candidate_gaussians']==0
    save(ROOT/'reports/EXTRA_VALIDATION.json',dict(passed=True,inference_mode_cases=checks,snapshot_independent=True,
        rng_unchanged=True,source_state_unchanged=True,finite_checks=True,empty_candidates=True))
    print('EXTRA_CHECKS_PASS',flush=True)


if __name__=='__main__':main()
