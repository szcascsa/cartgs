from common import *
from deployment_runtime import deployment_extension
from frozen_inference_cuda import extension


@torch.no_grad()
def main():
    ext=deployment_extension();original=extension();generator=torch.Generator(device='cuda').manual_seed(187)
    checks=0
    for n in [0,1,7,31,32,33,255,256,257,1023,4095,4096,4097,13301,16384,16385,32769,65536,65537,131073]:
        # Nontrivial row mapping verifies the fused eligibility gather and ID publication.
        rows=torch.randperm(max(1,2*n),generator=generator,device='cuda')[:n].sort().values
        eligible=torch.rand(max(1,2*n),generator=generator,device='cuda')>.2
        mask=eligible[rows];available=int(mask.sum())
        scratch=torch.empty(2+2*((min(n,65536)+255)//256),dtype=torch.int32,device='cuda')
        patterns=[torch.rand(n,generator=generator,device='cuda'),torch.zeros(n,device='cuda'),
            torch.ones(n,device='cuda'),torch.randint(0,8,(n,),generator=generator,device='cuda').float()/7,
            torch.linspace(-1e20,1e20,n,device='cuda')]
        if n:
            patterns[1][::2]=-0.;patterns[-1][::3]=torch.finfo(torch.float32).tiny
        for scores in patterns:
            for k in sorted(set([0,min(1,available),available//100,available//20,available//2,available])):
                x=rows[original.exact_select(scores,mask,k,[],True)]
                y=ext.select(scores,rows,eligible,k,scratch,[])
                assert torch.equal(x,y),(n,k,checks,x[:10],y[:10])
                assert len(y)==k and torch.equal(y,y.sort().values)
                checks+=1
        print('SELECTION_VALIDATED',n,checks,flush=True)
    save(ROOT/'reports/SELECTION_VALIDATION.json',dict(passed=True,checks=checks,N_max=131073,
        ties=True,signed_zero=True,eligibility=True,nontrivial_rows=True,large_map_cub=True))


if __name__=='__main__':main()
