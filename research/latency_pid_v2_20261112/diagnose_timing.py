"""同一次渲染同时测旧外层区间和原生图提交区间；输出逐位验收。"""
from pathlib import Path
import csv
import importlib.util
import json
import time

ROOT=Path(__file__).resolve().parent
prior=ROOT.parent/'latency_pid_native1200_target02_20261112/benchmark.py'
spec=importlib.util.spec_from_file_location('pid_old_benchmark',prior)
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
h,p,b,torch,np=old.h,old.p,old.b,old.torch,old.np
import timer_runtime


@torch.no_grad()
def main():
    h.configure();torch.set_num_threads(2);torch.manual_seed(0)
    scene,source=p.load_scene(b.prior.b.prior.SLUG)
    student,_,_=p.student_for(source,b.prior.b.prior.SLUG)
    engine=h.engine(source,student,'D');camera=h.Camera(source,'D',old.h.configure()[3],(1200,680))
    raster=timer_runtime.Renderer(40)
    spec=importlib.util.spec_from_file_location('pid_reference_raster',b.MODE40/'render_runtime.py')
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    binary=b.MODE40/'build/mycartgs_footprint_ninehour_v1.so'
    ref.extension=lambda:h.verified('mycartgs_footprint_ninehour_v1',binary,
        '4de72de81cbfa1eaaee6d065ce643c57ee875be29b34c690891baf389232656e')
    reference=ref.Renderer(40)
    events=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
    for event in events:event.record()
    torch.cuda.synchronize()
    a,s,e,z=events;rows=[];parity=[]
    frames=[0,200,500,722,1000,1500,1999]
    for frame in frames:
        cam=camera.create(frame)
        for budget in [256,512,1024,2000,3000,4000,8192]:
            packet,ids,stats=engine.prepare(cam,budget=budget)
            ir,rr=reference.render(cam,packet,source.scene.background)
            im,rd=raster.render(cam,packet,source.scene.background)
            same=bool(torch.equal(ir,im) and torch.equal(rr,rd))
            parity.append(dict(frame=frame,budget=budget,K=len(ids),bitwise=same))
            assert same
    torch.cuda.synchronize()
    for repeat in range(8):
        for frame in frames:
            cam=camera.create(frame)
            for budget in p.rotate([256,1024,3000,8192],repeat+frame):
                packet,ids,stats=engine.prepare(cam,budget=budget)
                t=time.perf_counter_ns();a.record()
                image,radii=raster.render(cam,packet,source.scene.background,(s.cuda_event,e.cuda_event))
                z.record();host_us=(time.perf_counter_ns()-t)/1000.
                z.synchronize()  # 独立诊断允许等待；正式控制不使用此等待。
                rows.append(dict(repeat=repeat,frame=frame,K=len(ids),outer_ms=a.elapsed_time(z),
                    inner_ms=s.elapsed_time(e),host_us=host_us))
    p.write_csv(ROOT/'reports/TIMING_BOUNDARY.csv',rows)
    p.save(ROOT/'reports/TIMER_BITWISE.json',parity)
    print('TIMING_DIAGNOSIS',json.dumps(dict(samples=len(rows),bitwise=len(parity),
        outer_p50=float(np.median([r['outer_ms'] for r in rows])),
        inner_p50=float(np.median([r['inner_ms'] for r in rows])),
        gap_p50=float(np.median([r['outer_ms']-r['inner_ms'] for r in rows])))),flush=True)


if __name__=='__main__':main()
