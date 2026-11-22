"""同帧输入比较P3逐Anchor、P4逐Anchor和P4逐点，检测新增分支的资源代价。"""
from prep_study import *
from native_camera import NativeCameraPreparer
from torch.utils.cpp_extension import _import_module_from_library
import gc


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--round',type=int,default=1);args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);rows=[]
    oldapi=runtime('preparation_p3_runtime',HERE/'development_p3/source/deployment_runtime.py')
    oldext=_import_module_from_library('mycartgs_tensor_preparation_v1',str(HERE/'development_p3'),True)
    oldapi.deployment_extension=lambda:oldext
    for slug in TUNE if args.round==1 else list(reversed(TUNE)):
        _,source=load_scene(slug);student,_,_=student_for(source,slug)
        old=oldapi.DeploymentRenderer(source.scene.model,source.scene.gaussians,source.scene.config,
            student=student,tensor_selector=2,tensor_compensator=2,tensor_warps=4,tensor_compensator_warps=2)
        old._pipeline._preparation_options(1,1,1)
        engines={'P3_anchor':old,'P4_anchor':make_engine(source,student,frustum=1,projection=1,device=1),
            'P4_point':make_engine(source,student,frustum=1,projection=2,device=1)}
        factory=NativeCameraPreparer(MiniCam);n=len(source.scene.views)
        frames=[int(.43*n),int(.43*n)+1,int(.43*n)+2,int(.07*n),int(.71*n),int(.97*n)]
        if args.round==2:frames.reverse()
        for ratio in PRIMARY:
            for frame in frames:
                cam=factory.from_source(source,frame);a,ia,_=old.prepare(cam,ratio_budget=ratio)
                for e in engines.values():
                    b,ib,_=e.prepare(cam,ratio_budget=ratio)
                    assert torch.equal(ia,ib) and all(bits_equal(v,b['inputs'][k]) for k,v in a['inputs'].items() if isinstance(v,torch.Tensor))
            renderer=create_renderer(factory.from_source(source,frames[0]),source.scene.background,old.total,use_graph=True)
            for repeat in range(20):
                for vi,frame in enumerate(frames):
                    for name in rotate(list(engines),repeat+vi+args.round):
                        e=engines[name]
                        for scope in ['engine_prepare','prepare','pipeline']:
                            cam=factory.from_source(source,frame);torch.cuda.synchronize();tick=time.perf_counter_ns()
                            if scope!='engine_prepare':cam=factory.from_source(source,frame)
                            packet,ids,stats=e.prepare(cam,ratio_budget=ratio)
                            if scope=='pipeline':renderer.render(cam,packet,source.scene.background)
                            torch.cuda.synchronize();ms=(time.perf_counter_ns()-tick)/1e6
                            rows.append(dict(scene=slug,ratio=ratio,scope=scope,variant=name,frame=frame,repeat=repeat,round=args.round,ms=ms))
            del renderer
            print('PROJECTION_VERSION_PAIR',slug,ratio,len(rows),flush=True)
        write_csv(out/'samples.csv',rows);del source,student,old,engines;gc.collect();torch.cuda.empty_cache()
    save(out/'SUMMARY.json',dict(passed=True,samples=len(rows),round=args.round,bitwise_equal=True))


if __name__=='__main__':main()
