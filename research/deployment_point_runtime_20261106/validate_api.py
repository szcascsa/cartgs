from common import *
from deployment_runtime import DeploymentRenderer


@torch.no_grad()
def main():
    tests=[]
    for group in ['both_li','selector_li','compensator_li','mode','legacy']:
        s=scene(group);training=s.model.training
        state={n:t.clone() for n,t in s.model.state_dict().items()}
        master={n:getattr(s.gaussians,n).clone() for n in ['_xyz','_opacity','_rotation','_scaling','_features_dc','_features_rest']}
        rng=torch.cuda.get_rng_state().clone()
        with torch.inference_mode():runtime=DeploymentRenderer(s.model,s.gaussians,s.config)
        ref=reference_renderer(s)
        for frame in [0,396,791]:
            cam=camera(s,frame)
            for request in [dict(budget=k) for k in [0,1,7,1000,20000]]+[dict(ratio_budget=r) for r in [0.,.01,.05,.1,.7,1.]]:
                for comp in [False,True]:
                    local=group!='legacy'
                    a,ia,sa=ref.prepare(cam,**request,view_conditioned=local,compensation=comp,measure_time=False)
                    b,ib,sb=runtime.prepare(cam,**request,view_conditioned=local,compensation=comp)
                    assert len(ia)==len(ib) and sa['candidate_gaussians']==sb['candidate_gaussians']
                    assert all(torch.isfinite(v).all() for v in b['inputs'].values() if isinstance(v,torch.Tensor))
                    tests.append(dict(group=group,frame=frame,request=request,compensation=comp,passed=True))
        cam=camera(s,396);local=group!='legacy'
        data,ids,_=runtime.prepare(cam,budget=1000,view_conditioned=local)
        saved={n:v.clone() for n,v in data['inputs'].items() if isinstance(v,torch.Tensor)}
        runtime.prepare(camera(s,0),ratio_budget=.7,view_conditioned=local)
        assert all(torch.equal(data['inputs'][n],v) for n,v in saved.items())
        # The deployment snapshot remains independent of later source-model updates.
        s.model.soft_net.weight.add_(.1);s.gaussians._xyz.add_(.01)
        again,chosen,_=runtime.prepare(cam,budget=1000,view_conditioned=local)
        assert torch.equal(ids,chosen) and all(torch.equal(again['inputs'][n],v) for n,v in saved.items())
        for n,t in s.model.state_dict().items():t.copy_(state[n])
        for n,t in master.items():getattr(s.gaussians,n).copy_(t)
        assert s.model.training==training and torch.equal(torch.cuda.get_rng_state(),rng)
        assert all(torch.equal(t,state[n]) for n,t in s.model.state_dict().items())
        assert all(p.grad is None for p in s.model.parameters())
        full,full_ids=runtime.full()
        for n,t in s.full_data['inputs'].items():
            if isinstance(t,torch.Tensor):assert torch.equal(t,full['inputs'][n]),(group,n)
        assert torch.equal(s.full_ids,full_ids)
        if group=='both_li':
            with torch.inference_mode():
                data,chosen,_=runtime.prepare(cam,budget=1000)
                assert torch.equal(chosen,ids)
                assert all(torch.equal(data['inputs'][n],v) for n,v in saved.items())
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                data,chosen,_=runtime.prepare(cam,budget=1000)
            torch.cuda.current_stream().wait_stream(stream)
            assert torch.equal(chosen,ids) and all(torch.equal(data['inputs'][n],v) for n,v in saved.items())
            for bad in [dict(budget=-1),dict(budget=True),dict(ratio_budget=float('nan')),dict(ratio_budget=-.1),dict(ratio_budget=1.1),dict(budget=1,ratio_budget=.1)]:
                try:runtime.prepare(cam,**bad)
                except ValueError:pass
                else:raise AssertionError(('Invalid input accepted',bad))
            no_points=camera(s,0)
            no_points.full_proj_transform=torch.eye(4,device='cuda');no_points.full_proj_transform[3,3]=-1e6
            empty,empty_ids,empty_stats=runtime.prepare(no_points,budget=1000)
            assert len(empty_ids)==0 and empty_stats['candidate_gaussians']==0
        with torch.enable_grad():
            try:runtime.prepare(cam,budget=1000)
            except RuntimeError as error:assert 'inference-only' in str(error)
            else:raise AssertionError('Training invocation was accepted')
        if group=='both_li':
            old=runtime._selector[0][0,0].clone();runtime._selector[0][0,0]=float('nan')
            try:runtime.prepare(cam,budget=1000)
            except RuntimeError as error:assert 'Nonfinite' in str(error)
            else:raise AssertionError('Selector finite check missing')
            runtime._selector[0][0,0].copy_(old)
            table=runtime._base.tables[0];old=table.clone();table.fill_(float('nan'))
            try:runtime.prepare(cam,budget=1000)
            except RuntimeError as error:assert 'Nonfinite' in str(error)
            else:raise AssertionError('Compensation finite check missing')
            table.copy_(old)
            loaded=DeploymentRenderer.from_checkpoint(CHECKPOINTS['both_li'])
            x,ix,_=loaded.prepare(cam,budget=1000);y,iy,_=runtime.prepare(cam,budget=1000)
            assert torch.equal(ix,iy)
            assert all(torch.equal(x['inputs'][n],v) for n,v in y['inputs'].items() if isinstance(v,torch.Tensor))
            save(ROOT/'reports/CHECKPOINT_LOAD.json',dict(passed=True,path=str(CHECKPOINTS['both_li'])))
        print('API',group,len(tests),flush=True)
        del runtime,ref,s;torch.cuda.empty_cache()
    save(ROOT/'reports/API_VALIDATION.json',dict(passed=True,cases=len(tests),tests=tests,
        independent_snapshot=True,source_model_unchanged=True,training_call_rejected=True,
        finite_checks=True,full_bitwise=True,output_ownership=True,checkpoint_reload=True))


if __name__=='__main__':main()
