from common import *
from baseline import previous_runtime
from deployment_runtime import DeploymentRenderer
from large_scene import LARGE


@torch.no_grad()
def main():
    repeat=int(sys.argv[1]);results=[]
    for group in ['tum','replica']:
        s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
        engines={'v27':previous_runtime(s)}
        for kind in [16,8208,4104]:
            run=DeploymentRenderer(s.model,s.gaussians,s.config)
            run._pipeline._small_tile_kind(kind);engines[str(kind)]=run
        for k in [128,511,512,768,1000,1024]:
            samples={key:[] for key in engines}
            for i in range(150):
                cam=camera(s,(i*73+19)%len(s.views));names=list(engines)
                shift=(i+repeat)%len(names);names=names[shift:]+names[:shift]
                if ((i+repeat)//len(names))%2:names.reverse()
                for name in names:
                    torch.cuda.synchronize();start=time.perf_counter()
                    packet,ids,stats=engines[name].prepare(cam,budget=k)
                    if i>=20:samples[name].append((time.perf_counter()-start)*1000)
            results.append(dict(group=group,k=k,median_ms={name:float(np.median(v)) for name,v in samples.items()},samples=samples))
            save(ROOT/'reports'/f'SMALL_ROUTE_BENCHMARK_{repeat}.json',dict(results=results))
            print(group,k,results[-1]['median_ms'],flush=True)
        del engines,s;torch.cuda.empty_cache()
    save(ROOT/'reports'/f'SMALL_ROUTE_BENCHMARK_{repeat}.json',dict(passed=True,results=results))


if __name__=='__main__':main()
