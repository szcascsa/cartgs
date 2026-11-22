"""只截取移动视角的热态prepare；不将Profiler耗时用作正式速度。"""
from study import *
@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--selector',type=int,required=True);p.add_argument('--compensator',type=int,required=True)
    p.add_argument('--warps',type=int,default=4);p.add_argument('--comp-warps',type=int,default=2);p.add_argument('--slug',default=TUNE[0]);p.add_argument('--ratio',type=float,default=.3);p.add_argument('--native',action='store_true');args=p.parse_args()
    _,source=load_scene(args.slug);student,_,_=student_for(source,args.slug)
    run=engine(source,student,args.selector,args.compensator,args.warps,original=not(args.selector or args.compensator),graph=not args.native,comp_warps=args.comp_warps)
    n=len(source.scene.views);frames=[int(.35*n)+j for j in range(3)]+[int(.63*n),int(.89*n),int(.11*n)]
    cams=[source.camera(f) for f in frames]
    for _ in range(3):
        for cam in cams:packet,ids,stats=run.prepare(cam,ratio_budget=args.ratio)
    torch.cuda.synchronize();torch.cuda.nvtx.range_push('TC_MEASURE')
    for j in range(36):
        torch.cuda.nvtx.range_push('tensor_prepare' if args.selector or args.compensator else 'baseline_prepare')
        packet,ids,stats=run.prepare(cams[j%6],ratio_budget=args.ratio);torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
    print('TC_PROFILE',args.slug,args.selector,args.compensator,args.ratio,len(ids),stats,flush=True)
if __name__=='__main__':main()
