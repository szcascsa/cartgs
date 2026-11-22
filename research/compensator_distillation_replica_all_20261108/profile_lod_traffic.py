"""NCU单帧访存范围：只读取现有checkpoint，不输出图像，不做速度结论。"""
from scene_runtime import *
import argparse
import gc
from deployment_runtime import DeploymentRenderer


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--slug',required=True)
    parser.add_argument('--mode',required=True);parser.add_argument('--scope',choices=['render','total'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--student',type=Path)
    args=parser.parse_args();scene=next(s for s in SCENES if s['slug']==args.slug)
    c,views=full_trajectory_views(Path(scene['checkpoint']),Path(scene['dataset']))
    frame=len(views)//2
    cam=MiniCam(c['width'],c['height'],focal2fov(c['fx'],c['width']),focal2fov(c['fy'],c['height']),
        np.linalg.inv(np.asarray(views[frame]['pose'])))
    background=torch.zeros(3,device='cuda')
    engine=DeploymentRenderer.from_checkpoint(scene['checkpoint'],student_checkpoint=args.student)
    def prepare():
        if args.mode=='full':
            packet,ids=engine.full();return packet,ids,{}
        return engine.prepare(cam,ratio_budget=float(args.mode))
    for _ in range(5):
        packet,ids,stats=prepare();result=render_prepared(cam,packet,background)
    torch.cuda.synchronize();del result
    if args.scope=='total':del packet,ids
    gc.collect();torch.cuda.synchronize()
    code=torch.cuda.cudart().cudaProfilerStart()
    assert int(code)==0,code
    if args.scope=='total':packet,ids,stats=prepare()
    result=render_prepared(cam,packet,background)
    torch.cuda.synchronize()
    code=torch.cuda.cudart().cudaProfilerStop()
    assert int(code)==0,code
    save(args.output,dict(scene=scene['id'],map_N=scene['gaussians'],frame=frame,
        mode=args.mode,scope=args.scope,K=len(ids),candidate_N=stats.get('candidate_gaussians'),
        width=c['width'],height=c['height'],saved_images=0,shared_gpu=True,scope_iterations=1,
        student_checkpoint=None if args.student is None else str(args.student),
        student_sha256=None if args.student is None else sha(args.student),
        selection_sha256=hashlib.sha256(ids.cpu().contiguous().numpy().tobytes()).hexdigest(),
        renderer_unchanged=True,models_changed=False,timing_claimed=False))
    print('TRAFFIC_CAPTURE_COMPLETE',args.slug,args.mode,args.scope,len(ids),flush=True)


if __name__=='__main__':main()
