"""真实Teacher运行路径，定位首层及四Head MMA调度的净代价。"""
from joint_study import *
from pending_runtime import PendingRenderer
from teacher_runtime import TeacherTensorRenderer
from balanced_order import order


@torch.no_grad()
def main():
    cases=[];owners=[];mode=json.loads((HERE/'reports/ACCEPTED_RENDER.json').read_text())['mode']
    for slug in TUNE:
        spec,source=load_source(slug);owners.append(source);factory=NativeCameraPreparer(p.MiniCam);bg=source.scene.background
        engines={'reference_FP32_comp':PendingRenderer(source,None,mode,native_dispatch=True)}
        engines.update({f'{kind}_w{w}':TeacherTensorRenderer(source,mode,m,w) for kind,m in [('fp16',3),('tf32',4)] for w in [1,2,4]})
        views=motions(source,quick=True)[::3];cam=factory.from_source(source,views[0][1])
        for engine in engines.values():engine.precompile(cam,bg)
        for condition in ['local0.01','local0.1','local0.3','local0.5','full']:
            for _,frame in views:
                for engine in engines.values():
                    for _ in range(3):engine.render(factory.from_source(source,frame),bg,condition)
                cases.append((slug,source,factory,frame,condition,engines))
    torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
    for repeat in range(5):
        for i,(slug,source,factory,frame,condition,engines) in enumerate(cases):
            for name in order(list(engines),repeat+i,reverse=repeat%2==1):
                torch.cuda.nvtx.range_push('/'.join(['nine_joint',slug,'teacher',condition,name,'pipeline']))
                engines[name].render(factory.from_source(source,frame),source.scene.background,condition)
                torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop();print('TEACHER_COMPONENT_PROFILE_COMPLETE',len(cases)*5*7,flush=True)


if __name__=='__main__':main()
