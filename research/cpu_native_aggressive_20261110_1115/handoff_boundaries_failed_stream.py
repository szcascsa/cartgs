"""验证相机同包上传、非默认CUDA Stream与租约错误检测。"""
from cpu_study import *
from cpu_gpu_camera_handoff import CpuGpuCameraHandoff


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from renderer import render_prepared
    source=load_scene('Replica__office2__rgbd');ref=source['reference']
    engine=CpuRenderer(source['model'],source['pc'],ref.noise,6);handoff=CpuGpuCameraHandoff(engine)
    stream=torch.cuda.Stream();background=torch.zeros(3,device='cuda');stream.wait_stream(torch.cuda.current_stream());checks=[]
    for i,condition in enumerate(['local0.01','local0.08','local0.7','fixed0','fixed1000']):
        cam=camera(source['records'][int([.19,.51,.83][i%3]*(len(source['records'])-1))])
        kwargs={'ratio_budget':float(condition[5:])} if condition.startswith('local') else {'budget':int(condition[5:])}
        ticket=handoff.prepare(cam,**kwargs)
        with torch.cuda.stream(stream):
            packet=ticket.wait(stream);result=render_prepared(ticket.camera,packet,background,return_radii=True)
            image=result['render'];radii=result['radii'];ticket.finish(stream)
        stream.synchronize()
        cpu,ids,_=engine.prepare(cam,**kwargs)
        expected=render_prepared(cuda_camera(cam),cuda_packet(cpu),background,return_radii=True)
        assert torch.equal(image,expected['render']) and torch.equal(radii,expected['radii'])
        for operation in [ticket.wait,ticket.finish]:
            try:operation()
            except RuntimeError:pass
            else:raise AssertionError('Released ticket accepted')
        checks.append(dict(condition=condition,K=len(ids),different_stream=True,same_rgb=True,same_radii=True,released_rejected=True))
    handoff.synchronize();write_csv(out/'checks.csv',checks)
    save(out/'SUMMARY.json',dict(passed=True,checks=len(checks)));print('HANDOFF_STREAM_PASS',len(checks),flush=True)


if __name__=='__main__':main()
