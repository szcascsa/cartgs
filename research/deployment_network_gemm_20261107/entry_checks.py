from common import *
from deployment_runtime import DeploymentRenderer
from baseline import previous_runtime
from large_scene import LARGE


@torch.no_grad()
def main():
    checks=Comparisons('ENTRY_VALIDATION.json');cases=0
    for group in ['tum','replica']:
        s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
        source=CHECKPOINTS['both_li'] if group=='tum' else LARGE
        old=previous_runtime(s);new=DeploymentRenderer.from_checkpoint(source)
        off=DeploymentRenderer.from_checkpoint(source,selector_tiled=False,compensation_tile=4)
        for frame in [0,len(s.views)//2,len(s.views)-1]:
            for ratio in [.01,.1,.7]:
                expected,ids,_=old.prepare(camera(s,frame),ratio_budget=ratio)
                for engine in [new,off]:
                    actual,selected,stats=engine.prepare(camera(s,frame),ratio_budget=ratio)
                    checks.same(str((group,frame,ratio,'ids')),ids,selected);checks.packet(str((group,frame,ratio)),expected,actual)
                    if engine is off:assert not stats['selector_tiled'] and stats['compensation_tile'] in [0,4]
                    cases+=1
        del s,old,new,off;torch.cuda.empty_cache()
    checks.finish(cases=cases,from_checkpoint=True,rollback_switches=True);print('ENTRY_PASS',cases,flush=True)


if __name__=='__main__':main()
