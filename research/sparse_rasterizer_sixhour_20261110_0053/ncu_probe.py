"""Nsight Compute单核测量入口，供占用率/SFU/访存分析；不用于墙钟结论。"""
from harness import *
from runtime import SparseRenderer


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',default='C1345')
    parser.add_argument('--condition',default='local0.01');parser.add_argument('--scene',default='Replica__office2__rgbd')
    args=parser.parse_args();scene,source=load_scene(args.scene)
    configs={'C1345':dict(render_variant=1,fused_keys=True,strided_camera=True,build_variant=1),
        'C7_alpha2':dict(render_variant=3),'C8_alpha4':dict(render_variant=4)}
    engine=baseline_B() if args.variant=='B' else SparseRenderer(**configs[args.variant])
    frame=views_for(source)[-2][1];camera=source.camera(frame);packet,ids,stats=prepare(source,camera,frame,args.condition)
    reference=render_prepared(camera,packet,source.scene.background)['render'];torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    image,radii=engine.render(camera,packet,source.scene.background);torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    assert bits_equal(reference,image)
    print('NCU_PARITY_PASS',scene['id'],args.condition,args.variant,len(ids),flush=True)


if __name__=='__main__':main()
