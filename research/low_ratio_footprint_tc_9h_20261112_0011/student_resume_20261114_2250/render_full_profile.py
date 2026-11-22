"""无Student的迁移地图仅取Full属性做嵌套删点Profile，不执行Teacher。"""
from student_study import *
from encoded_sign_runtime import EncodedSignRenderer
from thin_dispatch_runtime import ThinDispatchRenderer
from balanced_order import order
from nine_study import nested_packet


@torch.no_grad()
def main():
    cases=[];owners=[]
    slugs=['TUM_RGBD__freiburg1_xyz__rgbd','TUM_RGBD__freiburg2_xyz__mono','VECtor__corridors-dolly__stereo']
    for slug in slugs:
        spec,source=load_source(slug);owners.append(source)
        full,_,_=p.request(source.runtime,None,'full');bg=source.scene.background
        runs={'F2':BaselineRenderer(),'old38':Renderer(38),'final182':EncodedSignRenderer(182),'thin':ThinDispatchRenderer(182)}
        cam=source.camera(motions(source,quick=True)[0][1])
        for engine in runs.values():engine.precompile(source.runtime.total,cam,bg)
        for ratio in [.1,.3,1.]:
            packet,_=nested_packet(full,ratio)
            for _,frame in motions(source,quick=True)[::3]:
                cam=source.camera(frame)
                for engine in runs.values():
                    for _ in range(3):engine.render(cam,packet,bg)
                cases.append((slug,ratio,cam,packet,bg,runs))
    torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
    for repeat in range(5):
        for i,(slug,ratio,cam,packet,bg,runs) in enumerate(cases):
            for name in order(list(runs),repeat+i,reverse=repeat%2==1):
                torch.cuda.nvtx.range_push('/'.join(['nine_render',slug,'nested','full' if ratio==1 else 'local'+str(ratio),name]))
                runs[name].render(cam,packet,bg);torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop();print('MIGRATION_FULL_PROFILE_DONE',len(cases),flush=True)


if __name__=='__main__':main()
