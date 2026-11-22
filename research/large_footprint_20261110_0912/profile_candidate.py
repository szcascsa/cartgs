"""变视角Nsight Graph节点归因，不把Profiler延迟用于性能结论。"""
from study_common import *
from footprint_runtime import Renderer


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--mode',type=int,required=True)
    parser.add_argument('--condition',default='local0.01');args=parser.parse_args();out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    _,source=load_scene('Replica__office2__rgbd');views=views_for(source,'tune')[-4:]
    engine=previous_renderer() if args.mode<0 else Renderer(args.mode)
    engine.precompile(source.runtime.total,source.camera(views[0][1]),source.scene.background)
    packets=[]
    for _,frame in views:
        camera=source.camera(frame);packet,ids,_=prepare(source,camera,frame,args.condition)
        ref=render_prepared(camera,packet,source.scene.background,return_radii=True);image,radii=engine.render(camera,packet,source.scene.background)
        assert bits_equal(image,ref['render']) and bits_equal(radii,ref['radii'])
        packets.append((camera,packet))
    for camera,packet in packets:
        for _ in range(4):image=engine.render(camera,packet,source.scene.background)[0]
    torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
    for index in range(24):
        camera,packet=packets[index%len(packets)]
        torch.cuda.nvtx.range_push(f'footprint/{args.mode}/{args.condition}')
        image=engine.render(camera,packet,source.scene.background)[0];torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    save_json(out/'SUMMARY.json',dict(passed=True,calls=24,mode=args.mode,condition=args.condition,profiling_only=True))
    print('FOOTPRINT_PROFILE_COMPLETE',args.mode,args.condition,flush=True)


if __name__=='__main__':main()
