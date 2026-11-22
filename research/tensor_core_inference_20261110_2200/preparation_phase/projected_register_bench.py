from prep_study import *
from native_camera import NativeCameraPreparer
import gc


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);rows=[];checks=[]
    for slug in TUNE:
        _,source=load_scene(slug);student,_,_=student_for(source,slug)
        engines={'WMMA':make_engine(source,student,frustum=1,projection=1,device=1)}
        for warps in [1,2,4]:
            run=make_engine(source,student)
            run._pipeline._tensor_options(2,2,4,warps)
            run._pipeline._preparation_options(1,3,1)
            engines[f'register{warps}']=run
        factory=NativeCameraPreparer(MiniCam);n=len(source.scene.views)
        frames=[int(.43*n),int(.43*n)+1,int(.09*n),int(.81*n)]
        for ratio in PRIMARY:
            for frame in frames:
                cam=factory.from_source(source,frame);a,ia,_=engines['WMMA'].prepare(cam,ratio_budget=ratio)
                for name,e in engines.items():
                    b,ib,_=e.prepare(cam,ratio_budget=ratio);assert torch.equal(ia,ib)
                    checks.append(dict(scene=slug,variant=name,ratio=ratio,frame=frame,same_attributes=all(bits_equal(v,b['inputs'][k]) for k,v in a['inputs'].items() if isinstance(v,torch.Tensor)),max_error=max(difference(a,b).values(),default=0.)))
            renderer=create_renderer(factory.from_source(source,frames[0]),source.scene.background,engines['WMMA'].total,use_graph=True)
            for repeat in range(15):
                for vi,frame in enumerate(frames):
                    for name in rotate(list(engines),repeat+vi):
                        for scope in ['engine_prepare','prepare','pipeline']:
                            cam=factory.from_source(source,frame);torch.cuda.synchronize();tick=time.perf_counter_ns()
                            if scope!='engine_prepare':cam=factory.from_source(source,frame)
                            packet,ids,stats=engines[name].prepare(cam,ratio_budget=ratio)
                            if scope=='pipeline':renderer.render(cam,packet,source.scene.background)
                            torch.cuda.synchronize();ms=(time.perf_counter_ns()-tick)/1e6
                            rows.append(dict(scene=slug,variant=name,ratio=ratio,scope=scope,repeat=repeat,frame=frame,ms=ms))
            print('PROJECTED_REGISTER_PAIR',slug,ratio,len(rows),flush=True);del renderer
        write_csv(out/'samples.csv',rows);write_csv(out/'checks.csv',checks)
        del engines,source,student,run;gc.collect();torch.cuda.empty_cache()
    save(out/'SUMMARY.json',dict(passed=True,samples=len(rows),checks=len(checks),binary_sha256=sha(dp.deployment_extension().__file__)))


if __name__=='__main__':main()
