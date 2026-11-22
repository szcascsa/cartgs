from common import *
from deployment_runtime import DeploymentRenderer
from smoke import Frame


@torch.no_grad()
def main():
    s=scene();r=DeploymentRenderer(s.model,s.gaussians,s.config);cam=camera(s,396)
    for request in [dict(budget=0),dict(budget=1),dict(budget=7),dict(ratio_budget=.01),dict(budget=2000)]:
        for comp in [False,True]:r.prepare(cam,**request,compensation=comp)
    r._reference.noise.view(torch.int32).fill_(-1)
    try:r.prepare(cam,budget=5000)
    except RuntimeError as error:assert 'Nonfinite deployment Selector' in str(error)
    else:raise AssertionError('finite guard missing')
    capacity=16384;n=9171;k=1234;ext=r._ext
    scores=torch.rand(capacity,device='cuda');scores[n:]=float('nan')
    rows=torch.arange(capacity,device='cuda');rows[n:]=99999999
    eligible=torch.ones(capacity,dtype=torch.bool,device='cuda')
    out=torch.empty(capacity,dtype=torch.int64,device='cuda')
    blocks=(capacity+255)//256;scratch=torch.empty(2+4*blocks,dtype=torch.int32,device='cuda')
    for bad in [False,True]:
        f=Frame(n=n,k=k,status=int(bad));dc=torch.tensor(list(bytes(f)),device='cuda',dtype=torch.uint8)
        scratch[:2]=torch.tensor([0,k],device='cuda',dtype=torch.int32)
        ext._select_device(scores,rows,eligible,out,dc,scratch)
    torch.cuda.synchronize();print('SANITIZER_CASES_PASS',flush=True)


if __name__=='__main__':main()
