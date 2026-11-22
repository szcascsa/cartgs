from components import *
from deployment_runtime import deployment_extension
from radix_test import timer
import statistics


@torch.no_grad()
def main():
    ext=deployment_extension();checks=Comparisons('COMPENSATION_UNROLL_VALIDATION.json')
    results=[]
    for group in ['tum','replica']:
        s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
        old=previous_runtime(s)
        for ratio in [.01,.1,.7]:
            ids,args=sample(s,old,len(s.views)//2,ratio)
            base=previous_extension()._compensate_tiled(*args,16) if len(ids)>=1024 else previous_extension().compensate(*args)
            functions={'v21':lambda:previous_extension()._compensate_tiled(*args,16) if len(ids)>=1024 else previous_extension().compensate(*args)}
            for tile in [16,208,408,416,808,816,6416]:
                actual=ext._compensate_tiled(*args,tile)
                for j in range(3):checks.same(f'{group}/{ratio}/{tile}/{j}',base[j],actual[j])
                functions[str(tile)]=lambda t=tile:ext._compensate_tiled(*args,t)
            for fn in functions.values():
                for _ in range(5):fn()
            values={name:[] for name in functions}
            for repeat in range(12):
                names=list(functions);names=names[repeat%len(names):]+names[:repeat%len(names)]
                for name in names:values[name].append(timer(functions[name],20))
            row=dict(group=group,ratio=ratio,K=len(ids),median_us={k:statistics.median(x) for k,x in values.items()},samples=values)
            results.append(row);save(ROOT/'reports/COMPENSATION_UNROLL_BENCHMARK.json',dict(results=results))
            checks.finish(group=group,ratio=ratio)
            print(group,ratio,len(ids),row['median_us'],flush=True)
        del old,s;torch.cuda.empty_cache()
    checks.finish(groups=2)


if __name__=='__main__':main()
