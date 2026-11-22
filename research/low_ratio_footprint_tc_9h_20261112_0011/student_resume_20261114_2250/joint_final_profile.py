"""代表性Student完整当前帧Profile，分离Host范围和真实Graph节点。"""
from joint_final_study import *
from balanced_order import order


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--group',type=int,default=0);args=parser.parse_args()
    specs=student_specs(tune=True) if args.group==0 else student_specs()[-2:]
    cases=[];owners=[]
    for spec in specs:
        source,student,_=load_case(spec);owners.append((source,student))
        networks,renders,variants,dispatch=final_engines(source,student)
        keep=['A_fp32_F2','B_fp32_new','C_tensor_best_F2','D_tensor_best_new','E_tensor_best_old38']
        variants={name:variants[name] for name in keep}
        dispatch={name:dispatch[name] for name in ['G_cpp_pose','H_old38','I_final','I_no_camera','I_no_finish','I_no_members','I_no_fused']}
        factory=networks['tensor_best'][1];bg=source.scene.background;views=motions(source,quick=True)[::3]
        cam=factory.from_source(source,views[0][1])
        for renderer in renders.values():renderer.precompile(source.runtime.total,cam,bg)
        for engine in dispatch.values():engine.precompile(cam,bg)
        def render(name,frame,condition):
            if name in dispatch:return dispatch_frame(name,dispatch[name],source,frame,bg,condition,factory)
            (network,camera),renderer=variants[name];camera=camera.from_source(source,frame)
            packet,_,_=p.request(network,camera,condition);return renderer.render(camera,packet,bg)
        for condition in ['local0.01','local0.1','local0.3','local0.5','full']:
            for _,frame in views:
                for name in list(variants)+list(dispatch):
                    for _ in range(3):render(name,frame,condition)
                cases.append((spec['slug'],condition,source,frame,factory,bg,variants,dispatch))
    torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
    for repeat in range(5):
        for index,(slug,condition,source,frame,factory,bg,variants,dispatch) in enumerate(cases):
            for name in order(list(variants)+list(dispatch),repeat+index,reverse=repeat%2==1):
                torch.cuda.nvtx.range_push('/'.join(['nine_joint',slug,'student',condition,name,'pipeline']))
                if name in dispatch:dispatch_frame(name,dispatch[name],source,frame,bg,condition,factory)
                else:
                    (network,camera),renderer=variants[name];camera=camera.from_source(source,frame)
                    packet,_,_=p.request(network,camera,condition);renderer.render(camera,packet,bg)
                torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop();print('FINAL_JOINT_PROFILE_COMPLETE',len(cases),flush=True)


if __name__=='__main__':main()
