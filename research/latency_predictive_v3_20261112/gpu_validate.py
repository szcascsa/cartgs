"""真实冻结模型固定K产品封装等价与新控制器生命周期验收。"""
from pathlib import Path
import json
import numpy as np
import benchmark as b
import predictive_runtime as r

ROOT=Path(__file__).resolve().parent
h,p,torch=b.h,b.p,b.torch


@torch.no_grad()
def main():
    _,_,_,dm=h.configure();torch.set_num_threads(2)
    scene,source=p.load_scene(b.b.prior.b.prior.SLUG)
    student,_,_=p.student_for(source,b.b.prior.b.prior.SLUG)
    engine=h.engine(source,student,'D');camera=h.Camera(source,'D',dm,(1200,680));camera.create(0)
    raster=b.timer_runtime.Renderer(40);results=[]
    for budget in [256,1000,3000,8192,12000,16000,24000,engine.total]:
        controlled=r.PredictiveLatencyRenderer(engine,raster,source.scene.background,camera.factory.projection,(1200,680),
            config=dict(initial_budget=budget,min_budget=budget,max_budget=budget))
        for frame in [0,722,1999]:
            cam=camera.create(frame);out=controlled.render(cam,camera_to_world=source.scene.views[frame]['pose'])
            packet,ids,stats=engine.prepare(cam,budget=budget)
            image,radii=raster.render(cam,packet,source.scene.background)
            assert torch.equal(out['ids'],ids) and torch.equal(out['render'],image) and torch.equal(out['radii'],radii)
            assert torch.isfinite(image).all() and not image.requires_grad
            controlled.controller.poll();status=controlled.controller.stats()
            assert status['predicted_available']==stats['eligible_candidates']
            results.append(dict(budget=budget,frame=frame,K=len(ids),ids_rgb_radii_bitwise=True,
                predicted_N=status['predicted_available'],actual_N=stats['eligible_candidates']))
        assert controlled.controller.stats()['dropped_samples']==0;controlled.close()
    e=r.extension();s=r.anchor_summary(engine);projection=camera.factory.projection
    c=e.AsyncPID(dict(target_ms=.2,max_budget=engine.total),0,8)
    r.configure(c,s,projection,1200,680);pose=np.asarray(source.scene.views[0]['pose']).ravel().tolist()
    rejected=False
    try:r.configure(c,s,projection,1200,680)
    except RuntimeError:rejected=True
    assert rejected
    c.request(pose,False,1.);token,a,z=c.start_native(100,100);c.finish(token,False)
    torch.cuda.synchronize();c.poll();assert c.stats()['stale_samples']==1
    c.request(pose,True,2.);assert c.stats()['epoch']==1
    c.set_target(.3);assert c.stats()['target_ms']==.3
    c.request(pose,False,3.);token=c.start(0,0);c.finish(token,True);torch.cuda.synchronize();c.poll()
    other=torch.cuda.Stream();rejected=False
    try:
        with torch.cuda.stream(other):c.request(pose,False,4.)
    except RuntimeError:rejected=True
    assert rejected
    c.close();c.close()
    result=dict(bitwise_cases=results,lifecycle=['immutable_map','cancel','reset','target','empty','stream_guard','close'],
        summary_bytes=s.nbytes,anchors=len(s),passed=True)
    (ROOT/'reports/GPU_VALIDATION.json').write_text(json.dumps(result,indent=2));print('GPU_VALIDATION_PASS',len(results),len(result['lifecycle']),flush=True)


if __name__=='__main__':main()
