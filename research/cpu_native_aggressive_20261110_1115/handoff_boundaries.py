"""验证非默认Stream上的数据交接，再由原Rasterizer的默认Stream渲染。"""
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
            packet=ticket.wait(stream)
            captured=dict(sh_degree=packet['sh_degree'],inputs={name:value.clone() if isinstance(value,torch.Tensor) else value for name,value in packet['inputs'].items()})
            captured_camera=copy.copy(ticket.camera)
            for name in ['world_view_transform','projection_matrix','full_proj_transform','camera_center']:
                setattr(captured_camera,name,getattr(ticket.camera,name).clone())
            ticket.finish(stream)
        stream.synchronize()
        cpu,ids,_=engine.prepare(cam,**kwargs)
        for name,value in cpu['inputs'].items():
            if isinstance(value,torch.Tensor):assert torch.equal(captured['inputs'][name].cpu(),value),(condition,name)
        for name in ['world_view_transform','projection_matrix','full_proj_transform','camera_center']:
            assert torch.equal(getattr(captured_camera,name).cpu(),getattr(cam,name)),(condition,name)
        # 原Rasterizer含默认Stream核，不能直接在自定义Stream上下文中安全调用。
        # 本轮只测试上传消费者的Stream语义，不修改原Rasterizer或训练。
        result=render_prepared(captured_camera,captured,background,return_radii=True)
        expected=render_prepared(cuda_camera(cam),cuda_packet(cpu),background,return_radii=True)
        assert torch.equal(result['render'],expected['render']) and torch.equal(result['radii'],expected['radii'])
        for operation in [ticket.wait,ticket.finish]:
            try:operation()
            except RuntimeError:pass
            else:raise AssertionError('Released ticket accepted')
        checks.append(dict(condition=condition,K=len(ids),different_stream_transfer=True,render_stream='default',same_attributes=True,same_rgb=True,same_radii=True,released_rejected=True))
    handoff.synchronize();write_csv(out/'checks.csv',checks)
    save(out/'SUMMARY.json',dict(passed=True,checks=len(checks)));print('HANDOFF_STREAM_PASS',len(checks),flush=True)


if __name__=='__main__':main()
