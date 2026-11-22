"""最终Q独立Nsight取证，包含每帧CPU位姿、相机图、网络和Rasterizer。"""
from harness import *
from final_benchmark import native_modules,digest
from graph_runtime import GraphRenderer
from camera_async_preparation import AsyncGraphCameraPreparer as GraphCameraPreparer


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--condition',default='local0.01');args=parser.parse_args()
    output=HERE/'reports'/args.output;output.mkdir(exist_ok=False)
    camera_config=json.loads((HERE/'reports/FROZEN_CAMERA_ASYNC.json').read_text())
    for name,value in camera_config['sources'].items():assert digest(HERE/name)==value
    config=json.loads((HERE/'reports/FROZEN_CANDIDATE.json').read_text());native_modules(config)
    scene,source=load_scene('Replica__office2__rgbd');views=views_for(source,'final')[-4:]
    factory=GraphCameraPreparer(MiniCam);g=GraphRenderer(**config['graph'])
    camera=factory.from_source(source,views[0][1]);g.precompile(source.runtime.total,camera,source.scene.background)
    for _,frame in views:
        camera=factory.from_source(source,frame);packet,ids,_=prepare(source,camera,frame,args.condition)
        image=g.render(camera,packet,source.scene.background)[0]
    torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
    for index in range(20):
        _,frame=views[index%len(views)];torch.cuda.nvtx.range_push('R/pipeline/'+args.condition)
        torch.cuda.nvtx.range_push('camera_pose_graph');camera=factory.from_source(source,frame);torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push('prepare_selector_compensator');packet,ids,_=prepare(source,camera,frame,args.condition);torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push('rasterizer');image=g.render(camera,packet,source.scene.background)[0];torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    write_json(output/'SUMMARY.json',dict(passed=True,calls=20,condition=args.condition,scene=scene['id'],
        frozen_camera_sources=camera_config,profiling_only=True,all_inputs_recomputed=True,images_saved=False))
    print('R_PROFILE_COMPLETE',20,flush=True)


if __name__=='__main__':main()
