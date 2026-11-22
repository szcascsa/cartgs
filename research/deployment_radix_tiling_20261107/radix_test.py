from common import *
from baseline import previous_extension
from deployment_runtime import deployment_extension
import statistics


def scratch(n):
    return [torch.empty(s,device='cuda',dtype=torch.int32) for s in [n,n,264,2*((n+255)//256)]]


def sorting(n):
    return [torch.empty(n,device='cuda',dtype=t) for t in [torch.float32,torch.float32,torch.int64,torch.int64]]+[
        torch.empty(32*n+4096,device='cuda',dtype=torch.uint8)]


def timer(fn,repeats=10):
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):fn()
    end.record();end.synchronize()
    return start.elapsed_time(end)*1000/repeats


@torch.no_grad()
def main():
    candidate=deployment_extension();base=previous_extension();torch.manual_seed(137)
    checks=0;bench=[]
    for n in [0,1,31,256,1025,4096,13298,16384,32768,49102,65536,131072,524288]:
        space=2*n+17;rows=torch.randperm(space,device='cuda')[:n].contiguous()
        eligible=torch.rand(space,device='cuda')>.13;valid=int(eligible[rows].sum())
        work=scratch(n);old=sorting(n);small=torch.empty(2+2*((n+255)//256),device='cuda',dtype=torch.int32)
        status=torch.zeros(1,device='cuda',dtype=torch.int32)
        for kind in ['uniform','normal','ties','all_equal','signed_zero','saturated','extreme']:
            x=torch.rand(n,device='cuda') if kind=='uniform' else torch.randn(n,device='cuda')
            if kind=='ties':x=torch.randint(-5,5,(n,),device='cuda').float()
            if kind=='all_equal':x.fill_(1)
            if kind=='signed_zero':x=torch.where(x<0,-torch.zeros_like(x),torch.zeros_like(x))
            if kind=='saturated':x=torch.sigmoid(x*100)
            if kind=='extreme':x=torch.where(x<0,-torch.finfo(torch.float32).max,torch.finfo(torch.float32).max)
            for k in sorted(set([0,min(valid,1),int(valid*.01),int(valid*.7),valid])):
                expected=base.select(x,rows,eligible,k,small,old)
                actual=candidate._radix_select(x,rows,eligible,k,work,status)
                assert torch.equal(expected,actual),(n,kind,k,expected[:20].cpu(),actual[:20].cpu())
                checks+=1
            if n>=4096 and kind in ['uniform','ties','all_equal','saturated']:
                for fraction in [.01,.7]:
                    k=int(valid*fraction)
                    functions={'v7':lambda:base.select(x,rows,eligible,k,small,old),
                        'shrinking':lambda:candidate._radix_select(x,rows,eligible,k,work,status)}
                    for fn in functions.values():
                        for _ in range(5):fn()
                    samples={t:[] for t in functions}
                    for repeat in range(6):
                        for t in (list(functions) if repeat%2 else list(functions)[::-1]):samples[t].append(timer(functions[t]))
                    bench.append(dict(n=n,kind=kind,ratio=fraction,k=k,us={t:statistics.median(v) for t,v in samples.items()},samples=samples,
                        iterations=int(work[2][5]),remaining=int(work[2][0])))
        save(ROOT/'reports/RADIX_TEST.json',dict(passed=False,checks=checks,benchmarks=bench,n_completed=n))
        print('RADIX',n,checks,flush=True)
    save(ROOT/'reports/RADIX_TEST.json',dict(passed=True,checks=checks,benchmarks=bench))


if __name__=='__main__':main()
