"""冻结推理的参数/RNG/grad只读性以及非计时缓存命中元数据。"""
from common import *


@torch.no_grad()
def main():
    test=Comparisons('PURITY_VALIDATION.json');s=scene();model,pc=s.model,s.gaussians
    state={n:t.clone() for n,t in model.state_dict().items()}
    master={n:getattr(pc,n).clone() for n in ['_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest']}
    rng=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone()
    for backend in ['cuda','cuda_graph']:
        e=FixedBudgetRenderer(model,pc,s.config,0,backend)
        for frame in [0,396,791]:
            for _ in range(4):e.prepare(camera(s,frame),2000,view_conditioned=True,measure_time=False)
        e.prepare(None,global_ratio=.1)
        _,_,stats=e.prepare(None,global_ratio=.1,measure_time=False)
        assert stats['cache_hit'] and not stats['stage_timing_enabled']
    test.same('CPU_RNG',torch.get_rng_state(),rng);test.same('CUDA_RNG',torch.cuda.get_rng_state(),cuda)
    for n,t in model.state_dict().items():test.same('model/'+n,t,state[n])
    for n,t in master.items():test.same('master/'+n,getattr(pc,n),t)
    assert all(p.grad is None for p in model.parameters())
    test.finish(model_unchanged=True,master_unchanged=True,rng_unchanged=True,parameter_grads_empty=True,cache_timing_metadata=True)


if __name__=='__main__':main()
