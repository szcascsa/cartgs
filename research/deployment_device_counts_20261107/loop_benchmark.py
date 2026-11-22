from common import *
from baseline import first_stage_runtime
from deployment_runtime import DeploymentRenderer
from benchmark import stats


@torch.no_grad()
def main():
    s=scene();engines={'capacity_grid':first_stage_runtime(s),'bounded_grid':DeploymentRenderer(s.model,s.gaussians,s.config)}
    result=[]
    for repeat in [1,2]:
        for ratio in [.01,.7]:
            values={tag:[] for tag in engines}
            for i in range(220):
                cam=camera(s,((i-20)*137)%792);tags=list(engines)
                if (i+repeat)%2:tags.reverse()
                outputs=[]
                for tag in tags:
                    torch.cuda.synchronize();start=time.perf_counter()
                    packet,ids,_=engines[tag].prepare(cam,ratio_budget=ratio)
                    outputs.append((packet,ids))
                    if i>=20:values[tag].append((time.perf_counter()-start)*1000)
                assert torch.equal(outputs[0][1],outputs[1][1])
                assert all(torch.equal(v.contiguous().view(torch.uint8),outputs[1][0]['inputs'][k].contiguous().view(torch.uint8))
                    for k,v in outputs[0][0]['inputs'].items() if isinstance(v,torch.Tensor))
            row=dict(repeat=repeat,ratio=ratio,frames=200,summary={k:stats(v) for k,v in values.items()},raw_ms=values)
            result.append(row);save(ROOT/'reports/LOOP_BENCHMARK.json',dict(results=result));print(json.dumps({k:v for k,v in row.items() if k!='raw_ms'}),flush=True)


if __name__=='__main__':main()
