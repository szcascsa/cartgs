from common import *
from deployment_runtime import deployment_extension
from frozen_inference_cuda import extension
from smoke import Frame
import ctypes


@torch.no_grad()
def main():
    ext=deployment_extension();original=extension();gen=torch.Generator(device='cuda').manual_seed(283)
    checks=0;large_checks=0
    for capacity in [1,31,256,4096,13301,16384]:
        rows=torch.randperm(2*capacity,generator=gen,device='cuda')[:capacity].sort().values
        eligible=torch.rand(2*capacity,generator=gen,device='cuda')>.2
        output=torch.empty(capacity,dtype=torch.int64,device='cuda')
        blocks=(capacity+255)//256
        scratch=torch.empty(2+4*blocks,dtype=torch.int32,device='cuda')
        for n in sorted(set([0,1,capacity//3,capacity-1,capacity])):
            available=int(eligible[rows[:n]].sum())
            for pattern in ['random','ties','zeros']:
                scores=torch.rand(capacity,generator=gen,device='cuda')
                if pattern=='ties':scores=(scores*4).floor()/3
                if pattern=='zeros':scores.zero_();scores[::2]=-0.
                scores[n:]=float('nan')
                for k in sorted(set([0,min(1,available),available//100,available//2,available])):
                    frame=Frame(n=n,available=available,k=k,budget=k)
                    dc=torch.tensor(list(bytes(frame)),dtype=torch.uint8,device='cuda')
                    scratch[:2]=torch.tensor([0,k],dtype=torch.int32,device='cuda');output.fill_(-17)
                    ext._select_device(scores,rows,eligible,output,dc,scratch)
                    selected=original.exact_select(scores[:n],eligible[rows[:n]],k,[],True)
                    expected=rows[:n][selected]
                    assert torch.equal(expected,output[:k]),(capacity,n,pattern,k,expected[:20],output[:20])
                    assert bool((output[k:]==-17).all()),(capacity,n,k,'tail changed')
                    checks+=1;large_checks+=capacity>16384
            # Poison scores must never reach selection after Selector signals failure.
            frame=Frame(n=n,k=available,status=1)
            dc=torch.tensor(list(bytes(frame)),dtype=torch.uint8,device='cuda');output.fill_(-17)
            ext._select_device(scores,rows,eligible,output,dc,scratch)
            assert bool((output==-17).all())
        print('DEVICE_SELECT',capacity,checks,flush=True)
        save(ROOT/'reports/DEVICE_SELECTION.json',dict(passed=True,checks=checks,large_checks=large_checks,
            capacity_max=capacity,poisoned_tail=True,status_guard=True))
    print('DEVICE_SELECTION_PASS',flush=True)


if __name__=='__main__':main()
