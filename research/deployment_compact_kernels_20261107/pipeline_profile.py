from common import *
from baseline import previous_runtime
from deployment_runtime import DeploymentRenderer
from large_scene import LARGE


@torch.no_grad()
def main():
    records=[]
    for group in ['tum','replica']:
        s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
        engines={'v21':previous_runtime(s),
                 'unroll':DeploymentRenderer(s.model,s.gaussians,s.config,compensation_variant=6416),
                 'selection':DeploymentRenderer(s.model,s.gaussians,s.config,fused_selection=True),
                 'candidates':DeploymentRenderer(s.model,s.gaussians,s.config,fused_candidates=True),
                 'combined':DeploymentRenderer(s.model,s.gaussians,s.config,compensation_variant=6416,fused_selection=True,fused_candidates=True)}
        for label,request in [('1000',dict(budget=1000)),('5000',dict(budget=5000)),('70pct',dict(ratio_budget=.7))]:
            for i in range(30):
                cam=camera(s,(i*73+19)%len(s.views))
                names=list(engines)
                names=names[i%len(names):]+names[:i%len(names)]
                if i%2:names.reverse()
                for name in names:
                    torch.cuda.synchronize()
                    if i>=10:torch.cuda.nvtx.range_push(f'prepare/{name}/{group}/{label}')
                    _,ids,stats=engines[name].prepare(cam,**request)
                    if i>=10:
                        torch.cuda.nvtx.range_pop()
                        records.append(dict(group=group,condition=label,engine=name,N=stats['candidate_gaussians'],K=len(ids)))
        del engines,s;torch.cuda.empty_cache()
    save(ROOT/'reports/ABLATION_PROFILE_REQUESTS.json',dict(requests=records))


if __name__=='__main__':main()
