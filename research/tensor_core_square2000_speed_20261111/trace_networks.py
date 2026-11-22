"""实际移动视角的网络Kernel归因；Profiler时间不作为正式加速。"""
from helpers import *


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--slug',required=True)
    parser.add_argument('--ratio',type=float,required=True);args=parser.parse_args()
    c,d,cm,dm=configure();torch.set_num_threads(2)
    scene,source=p.load_scene(args.slug);student,_,_=p.student_for(source,args.slug)
    run=engine(source,student,'D');factory=Camera(source,'D',dm)
    frames=frames_for(source);frames=[frames[i][1] for i in [0,1,2,12,14,17]]
    renderer=p.create_renderer(factory.create(frames[0]),source.scene.background,run.total,use_graph=True)
    for _ in range(3):
        for frame in frames:
            cam=factory.create(frame);packet,ids,stats=run.prepare(cam,ratio_budget=args.ratio)
            renderer.render(cam,packet,source.scene.background)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push('TC_SQUARE_MEASURE')
    for i in range(36):
        torch.cuda.nvtx.range_push('camera_and_network_prepare')
        cam=factory.create(frames[i%len(frames)]);packet,ids,stats=run.prepare(cam,ratio_budget=args.ratio)
        torch.cuda.nvtx.range_pop();torch.cuda.nvtx.range_push('render_only')
        renderer.render(cam,packet,source.scene.background);torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()
    print('PROFILE_COMPLETE',scene['id'],args.ratio,36,stats,flush=True)


if __name__=='__main__':main()
