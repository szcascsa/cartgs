from components import *


@torch.no_grad()
def main():
    ext=deployment_extension();test=Comparisons('REGISTER_VALIDATION.json');timing=[]
    for group in ['both_li','selector_li','compensator_li','mode','legacy','replica']:
        s=scene(group) if group!='replica' else BudgetScene(LARGE,None,5000,0,None,'pytorch');a=previous_runtime(s)
        for frame in [0,len(s.views)//2,len(s.views)-1]:
            for ratio in RATIOS+[1.]:
                rows,scores,ids,args,_,_=sample(s,a,frame,ratio)
                baseline=previous_extension()._compensate_tiled(*args,4) if len(ids)>=1024 else previous_extension().compensate(*args)
                for tile in [16]:
                    actual=ext._compensate_tiled(*args,tile)
                    for j in range(3):test.same(str((group,frame,ratio,tile,j)),baseline[j],actual[j])
                if frame==0 and ratio==.7:
                    for edge in [0,1,3,7,8,15,16,17,31,32,33,63,64,65,127,128,129,511,512,1023,1024]:
                        partial=list(args);partial[1]=ids[:edge];target=previous_extension().compensate(*partial)
                        for tile in [16]:
                            result=ext._compensate_tiled(*partial,tile)
                            for j in range(3):test.same(str((group,'edge',edge,tile,j)),target[j],result[j])
                if frame==len(s.views)//2 and ratio in [.01,.1,.7]:
                    calls={'v11':lambda:previous_extension()._compensate_tiled(*args,4) if len(ids)>=1024 else previous_extension().compensate(*args),
                        'register16':lambda:ext._compensate_tiled(*args,16)}
                    samples={t:[] for t in calls}
                    for fn in calls.values():
                        for _ in range(5):fn()
                    for i in range(16):
                        for tag in (list(calls) if i%2 else list(calls)[::-1]):samples[tag].append(timer(calls[tag],30))
                    timing.append(dict(group=group,frame=frame,ratio=ratio,N=len(rows),K=len(ids),
                        median_us={t:statistics.median(v) for t,v in samples.items()},samples=samples))
            test.finish(group_completed=group,frame=frame);save(ROOT/'reports/REGISTER_BENCHMARK.json',dict(results=timing))
            print('REGISTER',group,frame,len(test.checks),flush=True)
        del a,s;torch.cuda.empty_cache()
    test.finish(groups=6)


if __name__=='__main__':main()
