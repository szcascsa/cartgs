from components import *


@torch.no_grad()
def main():
    newer=deployment_extension()
    spec=importlib.util.spec_from_file_location('mycartgs_deployment_fp32_v9',ROOT/'reports/stage_tile1/mycartgs_deployment_fp32_v9.so')
    older=importlib.util.module_from_spec(spec);spec.loader.exec_module(older)
    result=[]
    for group in ['both_li','replica']:
        s=scene(group) if group!='replica' else BudgetScene(LARGE,None,5000,0,None,'pytorch');a=previous_runtime(s)
        for ratio in [.01,.1,.7]:
            rows,scores,ids,args,_,_=sample(s,a,len(s.views)//2,ratio)
            calls={'original':lambda:newer.compensate(*args),
                'shared4':lambda:older._compensate_tiled(*args,4),'shared8':lambda:older._compensate_tiled(*args,8),
                'interleaved4':lambda:newer._compensate_tiled(*args,4),'interleaved8':lambda:newer._compensate_tiled(*args,8)}
            times={tag:[] for tag in calls}
            for fn in calls.values():
                for _ in range(10):fn()
            rng=np.random.default_rng(731)
            for _ in range(24):
                for tag in rng.permutation(list(calls)):times[tag].append(timer(calls[tag],30))
            row=dict(group=group,ratio=ratio,K=len(ids),median_us={t:statistics.median(v) for t,v in times.items()},samples=times)
            result.append(row);save(ROOT/'reports/TILE_COMPARISON.json',dict(results=result));print(row['group'],row['ratio'],row['median_us'],flush=True)


if __name__=='__main__':main()
