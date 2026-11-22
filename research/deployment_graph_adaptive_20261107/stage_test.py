from components import *
from deployment_runtime import DeploymentRenderer


def bit_errors(a,b):
    return dict(bitwise=torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8)),
                maximum=float((a-b).abs().max()) if a.numel() else 0.)


@torch.no_grad()
def main():
    ext=deployment_extension()
    checks=Comparisons('STAGE_EXACT_VALIDATION.json')
    quality=[]
    timings=[]
    for group in ['both_li','replica']:
        s=scene() if group=='both_li' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
        baseline=previous_runtime(s)
        engines={'off':DeploymentRenderer(s.model,s.gaussians,s.config,saturated_selection=False,dynamic_compensation=False),
                 'selection':DeploymentRenderer(s.model,s.gaussians,s.config,saturated_selection=True,dynamic_compensation=False),
                 'dynamic':DeploymentRenderer(s.model,s.gaussians,s.config,saturated_selection=False,dynamic_compensation=True)}
        for frame in [0,len(s.views)//2,len(s.views)-1]:
            cam=camera(s,frame)
            for request in [dict(ratio_budget=r) for r in [0.,.01,.05,.1,.3,.7,1.]]+[dict(budget=k) for k in [0,1,1000,2000,5000,100000]]:
                expected,ids,_=baseline.prepare(cam,**request)
                for name,engine in engines.items():
                    actual,selected,_=engine.prepare(cam,**request)
                    label=str((group,frame,request,name))
                    checks.same(label+'/ids',ids,selected)
                    checks.packet(label,expected,actual)
            for ratio in [.01,.1,.7]:
                rows,scores,ids,args,_,_=sample(s,baseline,frame,ratio)
                old=previous_extension()._compensate_tiled(*args,16) if len(ids)>=1024 else previous_extension().compensate(*args)
                new=ext._compensate_projected(*args)
                cam=camera(s,frame)
                x=render_prepared(cam,baseline._packet(old[0],len(ids)),s.background)['render']
                y=render_prepared(cam,baseline._packet(new[0],len(ids)),s.background)['render']
                entry=dict(group=group,frame=frame,ratio=ratio,K=len(ids),active_anchors=len(torch.unique(baseline._base.owners[ids])),
                           attributes=bit_errors(old[0],new[0]),pixels=bit_errors(x,y),finite=int(new[2])==0)
                if group=='both_li':
                    truth=s.ground_truth(frame)
                    psnr=lambda z:-10*float(torch.log10((z.clamp(0,1)-truth).square().mean().clamp_min(1e-30)))
                    entry['psnr_difference']=psnr(y)-psnr(x)
                quality.append(entry)
                if frame==len(s.views)//2:
                    functions={'v13':lambda:previous_extension()._compensate_tiled(*args,16) if len(ids)>=1024 else previous_extension().compensate(*args),
                               'projected':lambda:ext._compensate_projected(*args)}
                    samples={name:[] for name in functions}
                    for function in functions.values():
                        for _ in range(5):function()
                    for repeat in range(12):
                        for name in (list(functions) if repeat%2 else list(functions)[::-1]):samples[name].append(timer(functions[name],20))
                    timings.append(dict(group=group,ratio=ratio,K=len(ids),median_us={name:statistics.median(values) for name,values in samples.items()},samples=samples))
            checks.finish(group=group,frame=frame)
            save(ROOT/'reports/PROJECTED_STAGE.json',dict(quality=quality,timing=timings))
            print('STAGE',group,frame,len(checks.checks),flush=True)
        measurements={name:[] for name in engines}
        for name in ['selection','dynamic']:
            for ratio in [.01,.1,.7,1.]:
                samples={'off':[],'candidate':[]}
                for i in range(120):
                    cam=camera(s,(i*73+19)%len(s.views))
                    for tag,engine in [('off',engines['off']),('candidate',engines[name])][::1 if i%2 else -1]:
                        torch.cuda.synchronize()
                        start=time.perf_counter()
                        engine.prepare(cam,ratio_budget=ratio)
                        torch.cuda.synchronize()
                        if i>=20:samples[tag].append((time.perf_counter()-start)*1000)
                measurements[name].append(dict(ratio=ratio,median_ms={tag:statistics.median(values) for tag,values in samples.items()},samples=samples))
        save(ROOT/'reports'/f'STAGE_BENCHMARK_{group}.json',measurements)
        del engines,baseline,s
        torch.cuda.empty_cache()
    checks.finish(groups=2)


if __name__=='__main__':main()
