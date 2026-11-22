"""相同原Rasterizer：独立归因原生交接收益，包含相机与属性H2D。"""
from common import *
from cpu_gpu_camera_handoff import CpuGpuCameraHandoff
import gc

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--round',type=int,default=1);args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from renderer import render_prepared
    rows=[];checks=[];mem=[]
    slugs=['Replica__office2__rgbd','Replica__room0__rgbd','TUM_RGBD__freiburg1_xyz__rgbd']
    if args.round%2==0:slugs.reverse()
    for slug in slugs:
        source=load_scene(slug);records=source['records'];frames=[int(f*(len(records)-1)) for f in [.19,.51,.83]];cams=[camera(records[i]) for i in frames]
        base=native(source,'baseline');deep=native(source,'combined');background=torch.zeros(3,device='cuda')
        handoffs=dict(baseline=CpuGpuCameraHandoff(base),core_only=CpuGpuCameraHandoff(deep),native_all=extension().NativeHandoff(deep.native,deep.sh_degree))
        for name,h in handoffs.items():mem.append(dict(scene=slug,variant=name,**h.bytes()))
        for ci,condition in enumerate(['local0.01','local0.05','local0.1','local0.7','local0.08','fixed0','fixed1000','global0.1']):
            kw=dict(ratio_budget=float(condition[5:])) if condition.startswith('local') else dict(global_ratio=float(condition[6:]),view_conditioned=False) if condition.startswith('global') else dict(budget=int(condition[5:]))
            # 三个同时在途的租约＋拒绝覆盖，图像与直接CPU输出上传相同。
            h=handoffs['native_all'];tickets=[h.prepare(cam,**kw) for cam in cams]
            try:h.prepare(cams[0],**kw)
            except RuntimeError:pass
            else:raise AssertionError('Overwrote live slot')
            for i,ticket in enumerate(tickets):
                packet=ticket.wait();result=render_prepared(ticket.camera,packet,background,return_radii=True)
                cpu,ids,_=request(deep,cams[i],condition);assert torch.equal(ids,ticket.ids)
                for name,value in cpu['inputs'].items():
                    if isinstance(value,torch.Tensor):assert torch.equal(value,packet['inputs'][name].cpu()),(condition,name)
                expected=render_prepared(cuda_camera(cams[i]),cuda_packet(cpu),background,return_radii=True)
                assert torch.equal(result['render'],expected['render']) and torch.equal(result['radii'],expected['radii'])
                ticket.finish()
                for operation in [ticket.wait,ticket.finish]:
                    try:operation()
                    except RuntimeError:pass
                    else:raise AssertionError('Released ticket accepted')
                checks.append(dict(scene=slug,condition=condition,frame=frames[i],K=len(ids),same_attributes=True,same_rgb=True,same_radii=True,lease_errors_rejected=True))
            torch.cuda.synchronize()
            # 短块轮转，丢弃切换后的预热，避免两个线程池短暂自旋影响归因。
            for block in range(3):
                for name in rotate(list(handoffs),block+ci+args.round):
                    h=handoffs[name]
                    for j in range(6):
                        t=h.prepare(cams[j%3],**kw);packet=t.wait();render_prepared(t.camera,packet,background);t.finish()
                    torch.cuda.synchronize()
                    for j in range(18):
                        torch.cuda.synchronize();tick=time.perf_counter_ns()
                        t=h.prepare(cams[j%3],**kw);queued=(time.perf_counter_ns()-tick)/1e6
                        packet=t.wait();render_prepared(t.camera,packet,background);t.finish();torch.cuda.synchronize()
                        rows.append(dict(scene=slug,round=args.round,variant=name,condition=condition,block=block,sample=j,frame=frames[j%3],K=len(t.ids),
                            total_ms=(time.perf_counter_ns()-tick)/1e6,host_prepare_queue_ms=queued))
            write_csv(out/'samples.csv',rows);write_csv(out/'checks.csv',checks);write_csv(out/'memory.csv',mem)
            print('NATIVE_HANDOFF',slug,condition,len(rows),flush=True)
        for h in handoffs.values():h.synchronize()
        del tickets,ticket,t,packet,result,expected,cpu,ids,h,handoffs,base,deep,source;gc.collect();torch.cuda.empty_cache()
    save(out/'SUMMARY.json',dict(passed=True,round=args.round,samples=len(rows),checks=len(checks),scope='全CPU准备＋本帧相机/属性H2D＋默认Stream原Rasterizer',binary_sha256=sha(extension().__file__)))

if __name__=='__main__':main()
