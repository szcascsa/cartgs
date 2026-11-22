"""真实冻结engine经Python产品封装，逐项对照原版等K选择和图像。"""
from pathlib import Path
import importlib.util
import json

import diagnose_timing as diag
import timer_runtime

ROOT=Path(__file__).resolve().parent
h,p,b,torch=diag.h,diag.p,diag.b,diag.torch


@torch.no_grad()
def main():
    _,_,_,dm=h.configure();torch.set_num_threads(2)
    scene,source=p.load_scene(b.prior.b.prior.SLUG)
    student,_,_=p.student_for(source,b.prior.b.prior.SLUG)
    engine=h.engine(source,student,'D');camera=h.Camera(source,'D',dm,(1200,680))
    spec=importlib.util.spec_from_file_location('pid_v2_api',ROOT/'source/latency_pid.py')
    api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    spec=importlib.util.spec_from_file_location('pid_wrapper_reference',b.MODE40/'render_runtime.py')
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    reference=ref.Renderer(40);raster=timer_runtime.Renderer(40);records=[]
    for budget in [256,512,1000,2000,3000,4000,6000,8192]:
        controlled=api.LatencyPIDRenderer(engine,raster,source.scene.background,
            config=dict(initial_budget=budget,min_budget=budget,max_budget=budget,kp=0.,ki=0.),
            build_directory=str(ROOT/'build'))
        for frame in [0,722,1999]:
            cam=camera.create(frame);result=controlled.render(cam,camera_to_world=source.scene.views[frame]['pose'])
            packet,ids,stats=engine.prepare(cam,budget=budget)
            image,radii=reference.render(cam,packet,source.scene.background)
            assert torch.equal(result['ids'],ids)
            assert torch.equal(result['render'],image) and torch.equal(result['radii'],radii)
            assert not result['render'].requires_grad and torch.isfinite(image).all()
            controlled.poll()
            records.append(dict(frame=frame,budget=budget,K=len(ids),ids_bitwise=True,rgb_bitwise=True,radii_bitwise=True))
        assert controlled.poll()['dropped_samples']==0
        controlled.close()
    (ROOT/'reports/WRAPPER_BITWISE.json').write_text(json.dumps(records,indent=2))
    print('WRAPPER_BITWISE_PASS',len(records),flush=True)


if __name__=='__main__':main()
