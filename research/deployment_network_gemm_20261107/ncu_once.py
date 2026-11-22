from components import *


@torch.no_grad()
def main():
    s=scene();a=previous_runtime(s);ext=deployment_extension()
    _,_,_,args,_,_=sample(s,a,len(s.views)//2,.7)
    for _ in range(10):ext._compensate_tiled(*args,16)
    torch.cuda.synchronize();print('NCU_COMPONENT_COMPLETE',flush=True)


if __name__=='__main__':main()
