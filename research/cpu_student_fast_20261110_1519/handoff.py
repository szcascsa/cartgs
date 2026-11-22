"""真实切换视角：CPU全准备/上传/原渲染，四代表场景双轮。"""
from common import *
import gc

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--round',type=int,required=True);p.add_argument('--variant',default='vector');args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from renderer import render_prepared
    rows=[];checks=[];mem=[];raster=[];background=torch.zeros(3,device='cuda')
    slugs=['Replica__office1__mono','Replica__office2__rgbd','Replica__room0__rgbd','Replica__room2__mono']
    if args.round%2==0:slugs.reverse()
    conditions=PRIMARY+['local0.15','local0.45','fixed1000','fixed5000','global0.1']
    for si,slug in enumerate(slugs):
        source=load_scene(slug);n=len(source['records']);frames=[int(f*(n-1)) for f in [.19,.51,.83]];cams=[camera(source['records'][i]) for i in frames];identities={}
        # A single native CPU pool stays resident. Variant order rotates across scenes/rounds.
        for variant in rotate(['baseline',args.variant],si+args.round):
            engine=native(source,variant);module=baseline_extension() if variant=='baseline' else extension();h=module.NativeHandoff(engine.native,engine.sh_degree);mem.append(dict(scene=slug,round=args.round,variant=variant,**h.bytes()))
            for condition in conditions:
                kw=dict(ratio_budget=float(condition[5:])) if condition.startswith('local') else dict(global_ratio=float(condition[6:]),view_conditioned=False) if condition.startswith('global') else dict(budget=int(condition[5:]))
                for i,cam in enumerate(cams):
                    ticket=h.prepare(cam,**kw);packet=ticket.wait();render=render_prepared(ticket.camera,packet,background,return_radii=True);assert torch.isfinite(render['render']).all()
                    ids_hash=hashlib.sha256(ticket.ids.numpy().tobytes()).hexdigest();key=(condition,i)
                    if key in identities:assert ids_hash==identities[key]
                    else:identities[key]=ids_hash
                    checks.append(dict(scene=slug,round=args.round,condition=condition,variant=variant,frame=frames[i],K=len(ticket.ids),same_ids=True,finite=True,ids_sha256=ids_hash));ticket.finish()
                torch.cuda.synchronize()
                for block in range(3):
                    for j in range(6):
                        ticket=h.prepare(cams[j%3],**kw);render_prepared(ticket.camera,ticket.wait(),background);ticket.finish()
                    torch.cuda.synchronize()
                    for j in range(24):
                        torch.cuda.synchronize();tick=time.perf_counter_ns();ticket=h.prepare(cams[j%3],**kw);queued=(time.perf_counter_ns()-tick)/1e6
                        render_prepared(ticket.camera,ticket.wait(),background);ticket.finish();torch.cuda.synchronize()
                        rows.append(dict(scene=slug,round=args.round,condition=condition,variant=variant,frame=frames[j%3],block=block,sample=j,K=len(ticket.ids),host_ms=queued,total_ms=(time.perf_counter_ns()-tick)/1e6))
                # Render-only CUDA Event diagnostics: network and H2D outside interval.
                for j in range(9):
                    ticket=h.prepare(cams[j%3],**kw);packet=ticket.wait();torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record();render_prepared(ticket.camera,packet,background);end.record();end.synchronize();ticket.finish()
                    raster.append(dict(scene=slug,round=args.round,condition=condition,variant=variant,frame=frames[j%3],sample=j,K=len(ticket.ids),gpu_event_ms=start.elapsed_time(end)))
                print('STUDENT_HANDOFF',args.round,slug,variant,condition,len(rows),flush=True)
            h.synchronize();del h,engine,ticket,packet,render;gc.collect();torch.cuda.empty_cache()
        del source;gc.collect()
        for name,data in [('samples',rows),('checks',checks),('memory',mem),('raster',raster)]:write_csv(out/(name+'.csv'),data)
    save(out/'SUMMARY.json',dict(passed=True,round=args.round,samples=len(rows),checks=len(checks),single_pool=True,raster_diagnostic_rows=len(raster),scope='CPU准备＋相机属性H2D＋原Rasterizer；学生几何差异可能改变光栅工作量'))

if __name__=='__main__':main()
