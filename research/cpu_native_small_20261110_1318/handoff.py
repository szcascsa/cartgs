"""同输入帧的原生CPU准备＋上传＋原渲染器；上一版同样使用原生交接。"""
from common import *
import gc

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--round',required=True,type=int);args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from renderer import render_prepared
    rows=[];checks=[];mem=[];slugs=['Replica__office2__rgbd','Replica__room0__rgbd','TUM_RGBD__freiburg1_xyz__rgbd']
    if args.round%2==0:slugs.reverse()
    background=torch.zeros(3,device='cuda')
    for slug in slugs:
        source=load_scene(slug);records=source['records'];frames=[int(f*(len(records)-1)) for f in [.19,.51,.83]];cams=[camera(records[i]) for i in frames]
        old=native(source,'baseline');new=native(source,'combined')
        hs=dict(baseline=baseline_extension().NativeHandoff(old.native,old.sh_degree),combined=extension().NativeHandoff(new.native,new.sh_degree))
        for name,h in hs.items():mem.append(dict(scene=slug,variant=name,**h.bytes()))
        for ci,condition in enumerate(['local0.01','local0.05','local0.1','local0.5','local0.7','local0.08','fixed0','fixed1000','fixed5000','global0.01','global0.1']):
            kw=dict(ratio_budget=float(condition[5:])) if condition.startswith('local') else dict(global_ratio=float(condition[6:]),view_conditioned=False) if condition.startswith('global') else dict(budget=int(condition[5:]))
            for i,cam in enumerate(cams):
                a=hs['baseline'].prepare(cam,**kw);b=hs['combined'].prepare(cam,**kw);pa=a.wait();pb=b.wait()
                expected=render_prepared(a.camera,pa,background,return_radii=True);actual=render_prepared(b.camera,pb,background,return_radii=True)
                assert torch.equal(a.ids,b.ids) and torch.equal(expected['render'],actual['render']) and torch.equal(expected['radii'],actual['radii']);a.finish();b.finish()
                checks.append(dict(scene=slug,condition=condition,frame=frames[i],K=len(b.ids),same_rgb=True,same_radii=True,same_ids=True))
            torch.cuda.synchronize()
            for block in range(3):
                for name in rotate(list(hs),block+ci+args.round):
                    h=hs[name]
                    for j in range(6):
                        t=h.prepare(cams[j%3],**kw);render_prepared(t.camera,t.wait(),background);t.finish()
                    torch.cuda.synchronize()
                    for j in range(18):
                        torch.cuda.synchronize();tick=time.perf_counter_ns();t=h.prepare(cams[j%3],**kw);queued=(time.perf_counter_ns()-tick)/1e6
                        render_prepared(t.camera,t.wait(),background);t.finish();torch.cuda.synchronize()
                        rows.append(dict(scene=slug,round=args.round,variant=name,condition=condition,block=block,sample=j,frame=frames[j%3],K=len(t.ids),total_ms=(time.perf_counter_ns()-tick)/1e6,host_prepare_queue_ms=queued))
            write_csv(out/'samples.csv',rows);write_csv(out/'checks.csv',checks);write_csv(out/'memory.csv',mem);print('SMALL_HANDOFF',slug,condition,len(rows),flush=True)
        for h in hs.values():h.synchronize()
        del source,old,new,hs,h,t,a,b,pa,pb,expected,actual;gc.collect();torch.cuda.empty_cache()
    save(out/'SUMMARY.json',dict(passed=True,round=args.round,samples=len(rows),checks=len(checks),scope='双方均原生交接，同一原Rasterizer',binary_sha256=sha(extension().__file__)))

if __name__=='__main__':main()
