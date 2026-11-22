from prep_study import *
from native_camera import NativeCameraPreparer
import gc


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--bench',action='store_true');args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);checks=[];rows=[]
    for slug in TUNE:
        _,source=load_scene(slug);student,_,_=student_for(source,slug)
        factory=NativeCameraPreparer(MiniCam);c=make_engine(source,student,'C')
        engines={'C':c,'P1':make_engine(source,student,frustum=1,projection=1,device=1),
                 'P2':make_engine(source,student,frustum=1,projection=2,device=1)}
        n=len(source.scene.views);frames=[int(.43*n),int(.43*n)+1,int(.09*n),int(.81*n)]
        for ratio in [.01,.1,.2,.3,.4,.5,.7]:
            for frame in frames:
                cam=factory.from_source(source,frame);a,ia,sa=engines['P1'].prepare(cam,ratio_budget=ratio)
                b,ib,sb=engines['P2'].prepare(cam,ratio_budget=ratio)
                assert torch.equal(ia,ib) and sa['candidate_gaussians']==sb['candidate_gaussians']
                equal=all(bits_equal(v,b['inputs'][k]) for k,v in a['inputs'].items() if isinstance(v,torch.Tensor))
                assert equal,(slug,ratio,frame,difference(a,b))
                checks.append(dict(scene=slug,ratio=ratio,frame=frame,K=len(ia),bitwise=True))
        if args.bench:
            renderer=create_renderer(factory.from_source(source,frames[0]),source.scene.background,c.total,use_graph=True)
            for ratio in PRIMARY:
                for repeat in range(15):
                    for vi,frame in enumerate(frames):
                        for name in rotate(list(engines),repeat+vi):
                            e=engines[name]
                            for scope in ['engine_prepare','prepare','pipeline']:
                                cam=factory.from_source(source,frame);torch.cuda.synchronize();tick=time.perf_counter_ns()
                                if scope!='engine_prepare':cam=factory.from_source(source,frame)
                                packet,ids,stats=e.prepare(cam,ratio_budget=ratio)
                                if scope=='pipeline':renderer.render(cam,packet,source.scene.background)
                                torch.cuda.synchronize();ms=(time.perf_counter_ns()-tick)/1e6
                                rows.append(dict(scene=slug,variant=name,ratio=ratio,scope=scope,frame=frame,repeat=repeat,ms=ms,N=e.total,K=len(ids)))
                print('INLINE_PROJECTION_BENCH',slug,ratio,len(rows),flush=True)
            del renderer
        write_csv(out/'checks.csv',checks)
        if rows:write_csv(out/'samples.csv',rows)
        del engines,c,source,student;gc.collect();torch.cuda.empty_cache()
    save(out/'SUMMARY.json',dict(passed=True,checks=len(checks),samples=len(rows),binary_sha256=sha(dp.deployment_extension().__file__)))


if __name__=='__main__':main()
