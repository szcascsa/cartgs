from components import *


def errors(a,b):
    same=torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))
    return dict(bitwise=same,max_abs=float((a-b).abs().max()) if a.numel() else 0.,
        rms=float((a-b).square().mean().sqrt()) if a.numel() else 0.)


def packet(a,flat,n):
    return a._packet(flat,n)


@torch.no_grad()
def main():
    ext=deployment_extension();out=[];timing=[]
    for group in ['both_li','selector_li','compensator_li','mode','legacy','replica']:
        s=scene(group) if group!='replica' else BudgetScene(LARGE,None,5000,0,None,'pytorch');a=previous_runtime(s)
        packed=[torch.cat([a._compensator[5+4*j] for j in range(4)],dim=1),
            torch.cat([a._compensator[6+4*j] for j in range(4)])]
        for frame in [0,len(s.views)//2,len(s.views)-1]:
            for ratio in RATIOS+[1.]:
                rows,scores,ids,args,_,_=sample(s,a,frame,ratio)
                target=previous_extension()._compensate_tiled(*args,4) if len(ids)>=1024 else previous_extension().compensate(*args)
                actual=ext._compensate_gemm(*args,packed)
                assert int(actual[2])==0
                cam=camera(s,frame)
                x=render_prepared(cam,packet(a,target[0],len(ids)),s.background)['render']
                y=render_prepared(cam,packet(a,actual[0],len(ids)),s.background)['render']
                row=dict(group=group,frame=frame,ratio=ratio,N=len(rows),K=len(ids),
                    attributes=errors(target[0],actual[0]),delta=errors(target[1],actual[1]),pixels=errors(x,y))
                if group!='replica':
                    gt=s.ground_truth(frame)
                    psnr=lambda z:-10*float(torch.log10((z.clamp(0,1)-gt).square().mean().clamp_min(1e-30)))
                    row.update(psnr_before=psnr(x),psnr_after=psnr(y),psnr_delta=psnr(y)-psnr(x))
                out.append(row)
                if frame==len(s.views)//2 and ratio in [.01,.1,.7]:
                    calls={'v11':lambda:previous_extension()._compensate_tiled(*args,4) if len(ids)>=1024 else previous_extension().compensate(*args),
                        'gemm':lambda:ext._compensate_gemm(*args,packed)}
                    times={t:[] for t in calls}
                    for fn in calls.values():
                        for _ in range(5):fn()
                    for i in range(12):
                        for tag in (list(calls) if i%2 else list(calls)[::-1]):times[tag].append(timer(calls[tag],20))
                    timing.append(dict(group=group,ratio=ratio,N=len(rows),K=len(ids),
                        median_us={t:statistics.median(v) for t,v in times.items()},samples=times))
            print('GEMM',group,frame,len(out),flush=True)
            save(ROOT/'reports/GEMM_TEST.json',dict(passed=False,rows=out,timing=timing))
        del a,s;torch.cuda.empty_cache()
    losses=[-v['psnr_delta'] for v in out if 'psnr_delta' in v]
    assert np.mean(losses)<=.01 and np.percentile(losses,99)<=.05 and max(losses)<=.1
    save(ROOT/'reports/GEMM_TEST.json',dict(passed=True,rows=out,timing=timing,
        psnr_loss_mean=float(np.mean(losses)),psnr_loss_p99=float(np.percentile(losses,99)),psnr_loss_max=max(losses)))


if __name__=='__main__':main()
