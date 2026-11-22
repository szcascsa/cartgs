from prep_study import *
from native_camera import NativeCameraPreparer


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['A','B','C','D'],required=True);p.add_argument('--slug',default=TUNE[0]);args=p.parse_args()
    _,source=load_scene(args.slug);student,_,_=student_for(source,args.slug)
    f=json.loads((HERE/'reports/FROZEN_JOINT_CANDIDATE.json').read_text())
    run=make_engine(source,student,args.variant,frustum=1 if args.variant=='D' else 0,projection=f['projection'] if args.variant=='D' else 0,device=1 if args.variant=='D' else 0)
    factory=NativeCameraPreparer(MiniCam) if args.variant=='D' else create_camera_preparer(MiniCam,variant='R')
    n=len(source.scene.views);frames=[int(.43*n),int(.43*n)+1,int(.43*n)+2,int(.07*n),int(.71*n),int(.97*n)]
    for _ in range(3):
        for frame in frames:run.prepare(factory.from_source(source,frame),ratio_budget=.3)
    torch.cuda.synchronize();torch.cuda.nvtx.range_push('PREPARATION_MEASURE')
    for repeat in range(36):
        torch.cuda.nvtx.range_push('current_camera');cam=factory.from_source(source,frames[repeat%6]);torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push('complete_engine_prepare');packet,ids,stats=run.prepare(cam,ratio_budget=.3);torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
    print('PREPARATION_PROFILE',args.variant,args.slug,stats,flush=True)


if __name__=='__main__':main()
