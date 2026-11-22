"""原生三槽位：错误恢复、Stream依赖、动态尺寸、租约与独立输出检查。"""
from common import *

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from renderer import render_prepared
    source=load_scene('Replica__office2__rgbd');engine=native(source,'student');h=extension().NativeHandoff(engine.native,engine.sh_degree)
    cams=[camera(source['records'][int(f*(len(source['records'])-1))]) for f in [.19,.51,.83]];checks=[];background=torch.zeros(3,device='cuda')
    for kwargs in [dict(ratio_budget=-.1),dict(ratio_budget=float('nan')),dict(budget=-1),dict(ratio_budget=.1,budget=5),{}]:
        try:h.prepare(cams[0],**kwargs)
        except RuntimeError:checks.append(dict(case='invalid_request_rejected',detail=str(kwargs),passed=True))
        else:raise AssertionError(kwargs)
    for field in ['camera_center','projection_matrix']:
        bad=copy.copy(cams[0]);v=getattr(bad,field).contiguous().clone();v.view(-1)[0]=float('nan');setattr(bad,field,v)
        assert not torch.isfinite(getattr(bad,field)).all(),field
        try:h.prepare(bad,ratio_budget=.1)
        except RuntimeError:checks.append(dict(case='nonfinite_camera_rejected',detail=field,passed=True))
        else:raise AssertionError(field)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    for i,budget in enumerate([0,1,3,7,8,9,31,32,33,63,64,65,255,256,257,1000,5000,100000]):
        cam=cams[i%3];ticket=h.prepare(cam,budget=budget)
        with torch.cuda.stream(stream):
            packet=ticket.wait();saved={key:value.clone() if isinstance(value,torch.Tensor) else value for key,value in packet['inputs'].items()}
            saved_camera=copy.copy(ticket.camera)
            for key in ['world_view_transform','projection_matrix','full_proj_transform','camera_center']:setattr(saved_camera,key,getattr(ticket.camera,key).clone())
            ticket.finish()
        stream.synchronize();cpu,ids,_=engine.prepare(cam,budget=budget)
        assert torch.equal(ids,ticket.ids)
        for name,value in saved.items():
            if isinstance(value,torch.Tensor):assert torch.equal(value.cpu(),cpu['inputs'][name]),(budget,name)
        result=render_prepared(saved_camera,dict(sh_degree=engine.sh_degree,inputs=saved),background,return_radii=True)
        expected=render_prepared(cuda_camera(cam),cuda_packet(cpu),background,return_radii=True)
        assert torch.equal(result['render'],expected['render']) and torch.equal(result['radii'],expected['radii'])
        checks.append(dict(case='stream_transfer_dynamic_K',detail=str(budget),passed=True))
    h.synchronize();write_csv(out/'checks.csv',checks);save(out/'SUMMARY.json',dict(passed=True,checks=len(checks),render_stream='default',consumer_stream='nondefault',binary_sha256=sha(extension().__file__)))
    print('NATIVE_HANDOFF_EDGES_PASS',len(checks),flush=True)

if __name__=='__main__':main()
