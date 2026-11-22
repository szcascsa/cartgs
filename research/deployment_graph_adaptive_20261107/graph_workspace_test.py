from common import *
from deployment_runtime import DeploymentRenderer
from large_scene import LARGE
import gc


@torch.no_grad()
def main():
    rows=[]
    for group in ['tum','replica']:
        s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
        torch.cuda.synchronize();before=torch.cuda.memory_allocated()
        runtime=DeploymentRenderer(s.model,s.gaussians,s.config)
        checkpoints=[]
        budgets=[1,1025,2049,4097,8193,16385,32769]
        for repeat in range(4):
            for local in [False,True]:
                for comp in [False,True]:
                    for index,budget in enumerate(budgets):
                        packet,ids,stats=runtime.prepare(camera(s,(repeat*73+index)%len(s.views)),
                            budget=budget,view_conditioned=local,compensation=comp)
                        assert stats['graph_count']<=28
            torch.cuda.synchronize()
            checkpoints.append(dict(repeat=repeat,graph_count=stats['graph_count'],
                retained=torch.cuda.memory_allocated()-before,reserved=torch.cuda.memory_reserved()))
        assert checkpoints[1]['retained']==checkpoints[3]['retained'],checkpoints
        preserved={k:v.clone() for k,v in packet['inputs'].items() if isinstance(v,torch.Tensor)}
        runtime._pipeline._graph_enabled(False)
        for key,value in preserved.items():assert torch.equal(value,packet['inputs'][key])
        rows.append(dict(group=group,checkpoints=checkpoints,stable=True,ownership_after_clear=True))
        save(ROOT/'reports/GRAPH_WORKSPACE.json',dict(passed=True,results=rows))
        del runtime,s,packet,ids,preserved
        gc.collect();torch.cuda.empty_cache()
        print('WORKSPACE',group,checkpoints,flush=True)


if __name__=='__main__':main()
