"""单条件Nsight捕获；区分Host准备、GPU Kernel和同步，不把采集扰动当收益。"""
from harness import *
from final_benchmark import native_modules
from runtime import SparseRenderer
from graph_runtime import GraphRenderer


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--backend',choices=['A','B','C','G'],default='G')
    parser.add_argument('--condition',default='local0.01');parser.add_argument('--scope',choices=['raster','pipeline'],default='raster');args=parser.parse_args()
    output=HERE/'reports'/args.output;output.mkdir(exist_ok=False)
    config=json.loads((HERE/'reports/FROZEN_CANDIDATE.json').read_text());native_modules(config)
    scene,source=load_scene('Replica__office2__rgbd');views=views_for(source,'final')[-4:]
    engines={'B':baseline_B(),'C':SparseRenderer(**config['regular']),'G':GraphRenderer(**config['graph'])}
    engines['G'].precompile(source.runtime.total,source.camera(views[0][1]),source.scene.background)
    packets=[]
    for motion,frame in views:
        cam=source.camera(frame);packet,ids,stats=prepare(source,cam,frame,args.condition);packets.append((frame,cam,packet))
    def render(cam,packet):
        return render_prepared(cam,packet,source.scene.background)['render'] if args.backend=='A' else engines[args.backend].render(cam,packet,source.scene.background)[0]
    for frame,cam,packet in packets:
        for _ in range(4):image=render(cam,packet)
    torch.cuda.synchronize();count=20
    torch.cuda.cudart().cudaProfilerStart()
    for index in range(count):
        frame,cam,packet=packets[index%len(packets)]
        torch.cuda.nvtx.range_push(args.backend+'/'+args.scope+'/'+args.condition)
        if args.scope=='pipeline':
            torch.cuda.nvtx.range_push('camera_pose');cam=source.camera(frame);torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push('prepare_selector_compensator');packet,ids,stats=prepare(source,cam,frame,args.condition);torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push('rasterizer');image=render(cam,packet);torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    write_json(output/'SUMMARY.json',dict(passed=True,scene=scene['id'],condition=args.condition,backend=args.backend,scope=args.scope,
        calls=count,frames=[row[0] for row in packets],profiling_only=True,recomputes_inputs=args.scope=='pipeline',images_saved=False))
    print('FROZEN_PROFILE_COMPLETE',args.backend,args.scope,args.condition,count,flush=True)


if __name__=='__main__':main()
