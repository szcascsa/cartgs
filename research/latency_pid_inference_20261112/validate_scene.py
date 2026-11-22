"""真实冻结模型的等K逐位检查与因果单次渲染回归。"""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'source'))
from latency_pid import LatencyPIDRenderer
PRIOR=ROOT.parent/'office2_dynamic04_mode40_20261112'
spec=importlib.util.spec_from_file_location('pid_scene_prior',PRIOR/'benchmark.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
h,p,b,torch,np=d.h,d.p,d.b,d.torch,d.np


def save(name,value):p.save(ROOT/'reports'/name,value)
def write(name,rows):p.write_csv(ROOT/'reports'/name,rows)


@torch.no_grad()
def main():
    c,config,cm,dm=h.configure();torch.set_num_threads(2);torch.manual_seed(0)
    before=json.loads((PRIOR/'with_ours_full/formal/PROTECTED_AFTER.json').read_text())
    assert all(h.sha(path)==sha for path,sha in before.items())
    save('PROTECTED_BEFORE.json',before)
    scene,source=p.load_scene(b.prior.b.prior.SLUG)
    student,student_path,payload=p.student_for(source,b.prior.b.prior.SLUG)
    engine=h.engine(source,student,'D')
    camera=h.Camera(source,'D',dm,b.prior.b.SIZE)
    binary=b.MODE40/'build/mycartgs_footprint_ninehour_v1.so'
    binary_sha='4de72de81cbfa1eaaee6d065ce643c57ee875be29b34c690891baf389232656e'
    spec=importlib.util.spec_from_file_location('pid_mode40_runtime',b.MODE40/'render_runtime.py')
    runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
    runtime.extension=lambda:h.verified('mycartgs_footprint_ninehour_v1',binary,binary_sha)
    render=runtime.Renderer(40)
    build=str(ROOT/'build')
    checks=[]
    for budget in [256,2000,4000,8192]:
        wrapped=LatencyPIDRenderer(engine,render,source.scene.background,
            config=dict(initial_budget=budget,min_budget=budget,max_budget=budget,kp=0.,ki=0.),build_directory=build)
        for frame in [0,500,722,1000,1500,1999]:
            cam=camera.create(frame)
            result=wrapped.render(cam,camera_to_world=source.scene.views[frame]['pose'])
            actual=result['render'].clone();radii=result['radii'].clone();ids=result['ids'].clone()
            packet,reference_ids,stats=engine.prepare(cam,budget=budget)
            reference,reference_radii=render.render(cam,packet,source.scene.background)
            assert torch.equal(ids,reference_ids) and p.bits_equal(actual,reference) and p.bits_equal(radii,reference_radii)
            checks.append(dict(frame=frame,budget=budget,K=len(ids),ids_bitwise=True,rgb_bitwise=True,radii_bitwise=True))
        wrapped.close()
    write('PAIRED_BITWISE.csv',checks)

    # Only past completed Event samples drive this traversal: no GT, oracle K or trial render.
    wrapped=LatencyPIDRenderer(engine,render,source.scene.background,build_directory=build)
    rows=[];feedback={};start=time.perf_counter()
    for frame in range(2000):
        cam=camera.create(frame)
        result=wrapped.render(cam,camera_to_world=source.scene.views[frame]['pose'])
        stats=wrapped.controller.stats()
        assert 0<=len(result['ids'])<=min(result['pid_requested_budget'],result['stats']['eligible_candidates'])
        if stats['last_feedback_frame']>=1:
            feedback[int(stats['last_feedback_frame'])-1]=dict(render_ms=stats['last_feedback_ms'],K=stats['last_feedback_K'])
        rows.append(dict(frame=frame,requested_K=result['pid_requested_budget'],K=len(result['ids']),
            M=result['stats']['candidate_gaussians'],ratio=result['stats']['ratio'],epoch=stats['epoch'],
            updates=stats['updates'],feedback_frame=stats['last_feedback_frame'],integral=stats['integral']))
        if frame%200==0:print('PID_PROGRESS',frame,stats,flush=True)
    torch.cuda.synchronize()  # Verification boundary only; not inside the controller or render loop.
    stats=wrapped.poll()
    feedback[int(stats['last_feedback_frame'])-1]=dict(render_ms=stats['last_feedback_ms'],K=stats['last_feedback_K'])
    for row in rows:
        measurement=feedback.get(row['frame'])
        if measurement:assert row['K']==measurement['K']
        row['render_ms']=measurement['render_ms'] if measurement else ''
    write('CAUSAL_TRAJECTORY.csv',rows)
    elapsed=time.perf_counter()-start
    save('CONTROLLER_FINAL.json',stats)
    wrapped.close()
    after={path:h.sha(path) for path in before}
    save('PROTECTED_AFTER.json',after)
    changed=[path for path in before if before[path]!=after[path]]
    result=dict(passed=not changed,changed=changed,protected_files=len(before),scene=scene['id'],
        width=b.prior.b.SIZE[0],height=b.prior.b.SIZE[1],frames=len(rows),bitwise_pairs=len(checks),
        same_K_ids_RGB_radii_bitwise=True,causal_single_render=True,trajectory_wall_s=elapsed,
        observations=len(feedback),controller=stats,benchmark_valid_without_external_load_audit=False,
        images_saved=0,production_changed=False,training=False)
    save('SCENE_VALIDATION.json',result)
    assert result['passed']
    print('SCENE_VALIDATION_PASS',json.dumps(result),flush=True)


if __name__=='__main__':main()
