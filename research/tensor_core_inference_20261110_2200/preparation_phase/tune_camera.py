"""网络C/FPS与相机R/H的2x2配对，分离相机/准备协同。"""
from prep_study import *
from native_camera import NativeCameraPreparer
import gc


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);rows=[];checks=[]
    for slug in TUNE:
        _,source=load_scene(slug);student,_,_=student_for(source,slug)
        r=create_camera_preparer(MiniCam,variant='R');h=NativeCameraPreparer(MiniCam)
        c=make_engine(source,student,'C');d=make_engine(source,student,frustum=1,projection=1,device=1)
        variants={'C_R':(c,r),'C_H':(c,h),'FPS_R':(d,r),'FPS_H':(d,h)}
        n=len(source.scene.views);frames=[int(.43*n),int(.43*n)+1,int(.09*n),int(.81*n)]
        for ratio in [.1,.2,.3,.4,.5]:
            for name,(e,factory) in variants.items():
                for frame in frames:e.prepare(factory.from_source(source,frame),ratio_budget=ratio)
            cam=r.from_source(source,frames[0]);renderer=create_renderer(cam,source.scene.background,c.total,use_graph=True)
            for repeat in range(12):
                for vi,frame in enumerate(frames):
                    for name in rotate(list(variants),repeat+vi):
                        e,factory=variants[name]
                        for scope in ['camera','prepare','pipeline']:
                            torch.cuda.synchronize();tick=time.perf_counter_ns();cam=factory.from_source(source,frame)
                            if scope!='camera':packet,ids,stats=e.prepare(cam,ratio_budget=ratio)
                            if scope=='pipeline':renderer.render(cam,packet,source.scene.background)
                            torch.cuda.synchronize();ms=(time.perf_counter_ns()-tick)/1e6
                            rows.append(dict(scene=slug,variant=name,ratio=ratio,scope=scope,frame=frame,repeat=repeat,ms=ms,N=e.total))
            print('NATIVE_CAMERA_TUNE',slug,ratio,len(rows),flush=True)
            del renderer
        write_csv(out/'samples.csv',rows)
        del source,student,c,d,r,h;gc.collect();torch.cuda.empty_cache()
    save(out/'SUMMARY.json',dict(passed=True,samples=len(rows),binary_sha256=sha(dp.deployment_extension().__file__)))


if __name__=='__main__':main()
