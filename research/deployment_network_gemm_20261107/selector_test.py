from components import *


@torch.no_grad()
def main():
    ext=deployment_extension();test=Comparisons('SELECTOR_VALIDATION.json');timing=[]
    for group in ['both_li','selector_li','compensator_li','mode','legacy','replica']:
        s=scene(group) if group!='replica' else BudgetScene(LARGE,None,5000,0,None,'pytorch');a=previous_runtime(s);b=a._base
        for frame in [0,len(s.views)//2,len(s.views)-1]:
            for ratio in RATIOS+[1.]:
                rows,scores,ids,args,_,_=sample(s,a,frame,ratio);view=args[7]
                for local in ([False,True] if b.joint_schema else [False]):
                    arguments=(b.selector_static,rows,b.owners,view,a._reference.noise,a._selector,ratio,local,b.joint_schema)
                    expected=previous_extension().selector(*arguments);actual=ext._selector_tiled(*arguments)
                    for j in range(3):test.same(str((group,frame,ratio,local,j)),expected[j],actual[j])
                if frame==0 and ratio==.7:
                    for n in [0,1,3,4,7,15,16,17,31,32,33,63,64,65,127,128,129,511,512,1023,1024]:
                        small=list(arguments);small[1]=rows[:n]
                        expected=previous_extension().selector(*small);actual=ext._selector_tiled(*small)
                        for j in range(3):test.same(str((group,'edge',n,j)),expected[j],actual[j])
                if frame==len(s.views)//2 and ratio in [.01,.1,.7]:
                    calls={'v11':lambda:previous_extension().selector(*arguments),'register16':lambda:ext._selector_tiled(*arguments)}
                    samples={t:[] for t in calls}
                    for fn in calls.values():
                        for _ in range(5):fn()
                    for i in range(16):
                        for tag in (list(calls) if i%2 else list(calls)[::-1]):samples[tag].append(timer(calls[tag],30))
                    timing.append(dict(group=group,frame=frame,ratio=ratio,N=len(rows),
                        median_us={t:statistics.median(v) for t,v in samples.items()},samples=samples))
            test.finish(group_completed=group,frame=frame);save(ROOT/'reports/SELECTOR_BENCHMARK.json',dict(results=timing))
            print('SELECTOR',group,frame,len(test.checks),flush=True)
        del a,s;torch.cuda.empty_cache()
    test.finish(groups=6)


if __name__=='__main__':main()
