"""当前帧全链NVTX；组件耗时只用于Nsight归因，不冒充独占墙钟。"""
from joint_study import *
from balanced_order import order
from pending_runtime import PendingRenderer


@torch.no_grad()
def main():
    cases=[];owners=[];mode=json.loads((HERE/'reports/ACCEPTED_RENDER.json').read_text())['mode']
    for slug in ['Replica__office2__rgbd','Replica__room0__rgbd','TUM_RGBD__freiburg1_xyz__rgbd','VECtor__corner-slow__stereo']:
        spec,source=load_source(slug);student,path,payload=student_for_spec(source,spec);owners.append(source)
        for kind,weights in [('teacher',None)]+([('student',student)] if student is not None else []):
            networks=network_variants(source,weights);factory=networks['tensor_best'][1]
            renders={'F2':BaselineRenderer(),'new':Renderer(mode)};views=motions(source,quick=True)[::3]
            cam=factory.from_source(source,views[0][1]);bg=source.scene.background
            for r in renders.values():r.precompile(source.runtime.total,cam,bg)
            variants=benchmark_variants(networks,renders)
            dispatch={'E_pending':PendingRenderer(source,weights,mode),'F_native':PendingRenderer(source,weights,mode,native_dispatch=True)}
            for r in dispatch.values():r.precompile(cam,bg)
            for condition in ['local0.01','local0.1','local0.3','local0.5','full']:
                for _,frame in views:
                    for name,((engine,_),renderer) in variants.items():
                        for _ in range(3):
                            cam=factory.from_source(source,frame);packet,ids,stats=p.request(engine,cam,condition)
                            renderer.render(cam,packet,bg)
                    for r in dispatch.values():
                        for _ in range(3):r.render(cam,bg,condition)
                    cases.append((slug,kind,condition,source,frame,factory,variants,dispatch))
    torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
    for repeat in range(5):
        for index,(slug,kind,condition,source,frame,factory,variants,dispatch) in enumerate(cases):
            for name in order(list(variants)+list(dispatch),repeat+index,reverse=repeat%2==1):
                torch.cuda.nvtx.range_push('/'.join(['nine_joint',slug,kind,condition,name,'pipeline']))
                cam=factory.from_source(source,frame)
                if name in dispatch:dispatch[name].render(cam,source.scene.background,condition)
                else:
                    (engine,_),renderer=variants[name]
                    packet,ids,stats=p.request(engine,cam,condition);renderer.render(cam,packet,source.scene.background)
                torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    print('JOINT_PROFILE_COMPLETE',len(cases)*5*7,flush=True)


if __name__=='__main__':main()
