from common import *
from deployment_runtime import DeploymentRenderer
from baseline import previous_runtime
from large_scene import LARGE


@torch.no_grad()
def main():
    checks=0
    for group in ['tum','replica']:
        s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
        old=previous_runtime(s);new=DeploymentRenderer.from_checkpoint(CHECKPOINTS['both_li'] if group=='tum' else LARGE)
        off=DeploymentRenderer.from_checkpoint(CHECKPOINTS['both_li'] if group=='tum' else LARGE,shrinking_radix=False,compensation_tile=0)
        for frame in [0,len(s.views)//2,len(s.views)-1]:
            for ratio in [.01,.1,.7]:
                expected,ids,_=old.prepare(camera(s,frame),ratio_budget=ratio)
                for engine in [new,off]:
                    actual,selected,stats=engine.prepare(camera(s,frame),ratio_budget=ratio)
                    assert torch.equal(ids,selected)
                    for key,t in expected['inputs'].items():
                        if isinstance(t,torch.Tensor):assert torch.equal(t.contiguous().reshape(-1).view(torch.uint8),actual['inputs'][key].contiguous().reshape(-1).view(torch.uint8))
                    if engine is off:assert not stats['shrinking_radix'] and not stats['compensation_tile']
                    checks+=1
        del s,old,new,off;torch.cuda.empty_cache()
    save(ROOT/'reports/ENTRY_VALIDATION.json',dict(passed=True,cases=checks,from_checkpoint=True,rollback_switches=True))
    print('ENTRY_PASS',checks,flush=True)


if __name__=='__main__':main()
