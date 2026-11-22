"""TUM/VECtor冻结地图准备可迁移性；没有配套学生时保留Teacher CUDA补偿，不混称学生TC收益。"""
from prep_study import *
from native_camera import NativeCameraPreparer
import gc

SLUGS=['TUM_RGBD__freiburg1_xyz__rgbd','TUM_RGBD__freiburg2_xyz__mono',
       'VECtor__corner-slow__stereo','VECtor__corridors-dolly__stereo']


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--round',type=int,required=True);args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);rows=[];checks=[]
    for slug in SLUGS if args.round==1 else list(reversed(SLUGS)):
        spec,source=load_scene(slug)
        c=make_engine(source,None,'C',compensator=0)
        d=make_engine(source,None,'D',frustum=1,projection=0,device=1,compensator=0)
        old=create_camera_preparer(MiniCam,variant='R');new=NativeCameraPreparer(MiniCam)
        engines={'C_teacher':(c,old),'D_preparation_teacher':(d,new)}
        n=len(source.scene.views);frames=[int(.43*n),int(.43*n)+1,int(.43*n)+2,int(.07*n),int(.71*n),int(.97*n)]
        if args.round==2:frames.reverse()
        renderer=create_renderer(old.from_source(source,frames[0]),source.scene.background,c.total,use_graph=True)
        for ratio in PRIMARY:
            for frame in frames:
                ca=old.from_source(source,frame);da=new.from_source(source,frame)
                a,ia,sa=c.prepare(ca,ratio_budget=ratio);b,ib,sb=d.prepare(da,ratio_budget=ratio)
                assert len(ia)==len(ib) and sa['candidate_gaussians']==sb['candidate_gaussians']
                assert torch.equal(ia,ib),(slug,ratio,frame,'selector IDs changed')
                ar=render_prepared(ca,a,source.scene.background,return_radii=True)['render'];br=render_prepared(da,b,source.scene.background,return_radii=True)['render']
                gt=source.scene.ground_truth(frame).clamp(0,1)
                apsnr=float(-10*torch.log10((ar.clamp(0,1)-gt).square().mean().clamp_min(1e-30)))
                bpsnr=float(-10*torch.log10((br.clamp(0,1)-gt).square().mean().clamp_min(1e-30)))
                checks.append(dict(scene=spec['id'],ratio=ratio,frame=frame,round=args.round,N=c.total,K=len(ia),M=sa['candidate_gaussians'],
                    same_ids=True,reference_psnr=apsnr,psnr=bpsnr,psnr_loss=apsnr-bpsnr,max_rgb_error=float((ar-br).abs().max())))
            for repeat in range(8):
                for vi,frame in enumerate(frames):
                    for name in rotate(list(engines),repeat+vi+args.round):
                        engine,factory=engines[name]
                        for scope in ['engine_prepare','prepare','pipeline']:
                            cam=factory.from_source(source,frame);torch.cuda.synchronize();tick=time.perf_counter_ns()
                            if scope!='engine_prepare':cam=factory.from_source(source,frame)
                            packet,ids,stats=engine.prepare(cam,ratio_budget=ratio)
                            if scope=='pipeline':renderer.render(cam,packet,source.scene.background)
                            torch.cuda.synchronize();ms=(time.perf_counter_ns()-tick)/1e6
                            rows.append(dict(scene=spec['id'],variant=name,ratio=ratio,scope=scope,round=args.round,repeat=repeat,frame=frame,N=c.total,K=len(ids),M=stats['candidate_gaussians'],ms=ms))
            print('PREPARATION_TRANSFER',spec['id'],ratio,len(rows),flush=True)
        write_csv(out/'samples.csv',rows);write_csv(out/'checks.csv',checks)
        del source,c,d,engines,renderer;gc.collect();torch.cuda.empty_cache()
    save(out/'SUMMARY.json',dict(passed=True,samples=len(rows),checks=len(checks),round=args.round,
        scope='只有准备和调度的迁移测试，Selector FP16，Compensator保持原64/64 Teacher CUDA，无学生预投影'))


if __name__=='__main__':main()
