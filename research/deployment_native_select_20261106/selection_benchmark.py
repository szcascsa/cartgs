from common import *
from deployment_runtime import deployment_extension
from frozen_inference_cuda import extension
import statistics


@torch.no_grad()
def main():
    ext=deployment_extension();original=extension();result=[]
    for n in [4096,13301,32768,65536,131072]:
        rows=torch.arange(n,device='cuda');eligible=torch.ones(n,dtype=torch.bool,device='cuda')
        scores=torch.rand(n,device='cuda');scratch=torch.empty(2+2*((min(n,65536)+255)//256),dtype=torch.int32,device='cuda')
        workspace=[torch.empty(n,device='cuda',dtype=t) for t in [torch.float32,torch.float32,torch.int64,torch.int64]]
        workspace.append(torch.empty(32*n+4096,device='cuda',dtype=torch.uint8))
        for ratio in [.01,.1,.7]:
            k=int(n*ratio)
            funcs={'cub':lambda:rows[original.exact_select(scores,eligible[rows],k,workspace,True)],
                'radix':lambda:ext.select(scores,rows,eligible,k,scratch,workspace)}
            values={name:[] for name in funcs}
            for repeat in range(9):
                for name in (list(funcs) if repeat%2 else list(funcs)[::-1]):
                    for _ in range(10):funcs[name]()
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(100):funcs[name]()
                    end.record();end.synchronize();values[name].append(start.elapsed_time(end)/100)
            row=dict(N=n,ratio=ratio,K=k,median_ms={name:statistics.median(v) for name,v in values.items()},raw=values)
            result.append(row);print(json.dumps({k:v for k,v in row.items() if k!='raw'}),flush=True)
            save(ROOT/'reports/SELECTION_BENCHMARK.json',dict(results=result))


if __name__=='__main__':main()
