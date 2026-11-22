"""同帧单次连续H2D的完整CPU准备→原GPU渲染，独立测量各阶段。"""
from cpu_study import *
from cpu_gpu_camera_handoff import CpuGpuCameraHandoff

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--student',action='store_true');args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    from renderer import render_prepared
    rows=[];checks=[];memory=[];components=[]
    for slug in ['Replica__office2__rgbd','Replica__room0__rgbd','TUM_RGBD__freiburg1_xyz__rgbd']:
        source=load_scene(slug);ref=source['reference'];student=load_student(source['student_path'],'cpu')[0] if args.student else None
        engine=CpuRenderer(source['model'],source['pc'],ref.noise,6,student);transfer=CpuGpuCameraHandoff(engine)
        memory.append(dict(scene=slug,cpu_workspace=engine.native.workspace_bytes(),**transfer.bytes()))
        records=source['records'];frames=[int(f*(len(records)-1)) for f in [.19,.51,.83]];cams=[camera(records[i]) for i in frames]
        gpu_cams=[cuda_camera(cam) for cam in cams];background=torch.zeros(3,device='cuda')
        for ratio in [.01,.1,.7]:
            for index in range(9):
                ticket=transfer.prepare(cams[index%3],ratio_budget=ratio);packet=ticket.wait()
                result=render_prepared(ticket.camera,packet,background,return_radii=True);ticket.finish()
            torch.cuda.synchronize()
            for repeat in range(2):
                for index in range(30):
                    i=index%3;cam=cams[i];gpu_cam=gpu_cams[i]
                    torch.cuda.synchronize();tick=time.perf_counter_ns()
                    ticket=transfer.prepare(cam,ratio_budget=ratio);host_queued=(time.perf_counter_ns()-tick)/1e6
                    packet=ticket.wait();result=render_prepared(ticket.camera,packet,background,return_radii=True);ticket.finish()
                    torch.cuda.synchronize();total=(time.perf_counter_ns()-tick)/1e6
                    rows.append(dict(scene=slug,ratio=ratio,repeat=repeat,frame=frames[i],M=ticket.stats['candidate_gaussians'],K=len(ticket.ids),
                        total_cpu_prepare_H2D_render_ms=total,host_prepare_and_queue_ms=host_queued,H2D_bytes=(len(ticket.ids)*engine.width+51)*4,student=args.student))
            # 计时外：显式租约回收前不得复用，连续提交三帧后比对，确保没有旧帧覆盖。
            tickets=[transfer.prepare(cam,ratio_budget=ratio) for cam in cams]
            try:transfer.prepare(cams[0],ratio_budget=ratio)
            except RuntimeError:pass
            else:raise AssertionError('Unreleased slot reused')
            rendered=[]
            for i,ticket in enumerate(tickets):
                packet=ticket.wait();result=render_prepared(ticket.camera,packet,background,return_radii=True);rendered.append(result['render']);ticket.finish()
            torch.cuda.synchronize()
            for i,image in enumerate(rendered):
                cpu_packet,ids,info=engine.prepare(cams[i],ratio_budget=ratio)
                reference=render_prepared(gpu_cams[i],cuda_packet(cpu_packet),background,return_radii=True)
                assert torch.equal(image,reference['render']),'Handoff changed native output'
                checks.append(dict(scene=slug,ratio=ratio,frame=frames[i],K=len(ids),same_pixels=True,lease_overwrite_rejected=True))
            for i,cam in enumerate(cams):
                host=torch.empty(engine.total*engine.width,pin_memory=True);device=torch.empty_like(host,device='cuda')
                for iteration in range(12):
                    tick=time.perf_counter_ns();flat,ids,info=engine.prepare(cam,ratio_budget=ratio,output=host,flat_output=True)
                    cpu_ms=(time.perf_counter_ns()-tick)/1e6
                    first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True)
                    first.record();device[:flat.numel()].copy_(flat,non_blocking=True);last.record();last.synchronize()
                    upload_ms=first.elapsed_time(last);packet=engine.packet(device[:flat.numel()],len(ids))
                    first.record();result=render_prepared(gpu_cams[i],packet,background,return_radii=True);last.record();last.synchronize()
                    components.append(dict(scene=slug,ratio=ratio,frame=frames[i],iteration=iteration,K=len(ids),cpu_prepare_ms=cpu_ms,
                        H2D_event_ms=upload_ms,raster_event_ms=first.elapsed_time(last),H2D_bytes=flat.numel()*4))
            write_csv(out/'samples.csv',rows);write_csv(out/'checks.csv',checks);write_csv(out/'components.csv',components)
            print('CPU_HANDOFF',slug,ratio,len(rows),flush=True)
        transfer.synchronize();del source,ref,engine,transfer,tickets,rendered,packet,result,reference,cpu_packet;torch.cuda.empty_cache()
    write_csv(out/'memory.csv',memory);save(out/'SUMMARY.json',dict(passed=True,samples=len(rows),checks=len(checks),student=args.student,
        prep='全CPU',GPU='一次连续H2D＋原Rasterizer',camera_pose_tensor_creation_excluded=True,
        copy_event_ownership_checked=True,no_same_frame_cache=True,binary_sha256=sha(extension().__file__)))

if __name__=='__main__':main()
