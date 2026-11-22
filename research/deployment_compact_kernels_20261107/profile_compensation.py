from components import *


@torch.no_grad()
def main():
    group=sys.argv[1] if len(sys.argv)>1 else 'replica'
    s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
    old=previous_runtime(s)
    ids,args=sample(s,old,len(s.views)//2,.7)
    for _ in range(3):previous_extension()._compensate_tiled(*args,16)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push('network_compensator')
    previous_extension()._compensate_tiled(*args,16)
    torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
    print('PROFILED',group,len(ids),flush=True)


if __name__=='__main__':main()
