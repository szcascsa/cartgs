"""V3：相同原生计时的V2/解除上限/视角预测/全部候选四方法。"""
from pathlib import Path
import argparse
import csv
import importlib.util
import json
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
PID_ROOT=ROOT.parent/'latency_pid_inference_20261112'
sys.path.insert(0,str(PID_ROOT/'source'))
from latency_pid import pid_extension as legacy_extension
PRIOR=ROOT.parent/'office2_dynamic04_mode40_20261112'
spec=importlib.util.spec_from_file_location('pid_shared_prior',PRIOR/'benchmark.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
h,p,b,torch,np=d.h,d.p,d.b,d.torch,d.np
import cv2
import predictive_runtime
V2_ROOT=ROOT.parent/'latency_pid_v2_20261112'
spec=importlib.util.spec_from_file_location('v3_native_timer',V2_ROOT/'timer_runtime.py')
timer_runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(timer_runtime)

def new_modules():
    binary=V2_ROOT/'build/mycartgs_latency_pid_v2.so'
    return dict(v2=h.verified('mycartgs_latency_pid_v2',binary,
        json.loads((V2_ROOT/'reports/FINAL_SERVER_INTEGRITY.json').read_text())['compiled_binaries']['build/mycartgs_latency_pid_v2.so']),
        v3=predictive_runtime.extension())

METHODS=['v2_8192','v2_expanded','v2_fast','predictive','all_candidates']
CONFIGS={}


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    name='smoke' if args.smoke else 'formal'
    output=ROOT/name;output.mkdir(exist_ok=False)
    save=lambda name,obj:p.save(output/name,obj)
    write=lambda name,rows:p.write_csv(output/name,rows)
    c,config,cm,dm=h.configure();torch.set_num_threads(2);torch.manual_seed(0)
    before=json.loads((PID_ROOT/'reports/PROTECTED_AFTER.json').read_text())
    for rel,digest in json.loads((V2_ROOT/'reports/SOURCE_SHA256.json').read_text()).items():before[str(V2_ROOT/rel)]=digest
    for rel,digest in json.loads((V2_ROOT/'reports/FINAL_SERVER_INTEGRITY.json').read_text())['compiled_binaries'].items():before[str(V2_ROOT/rel)]=digest
    for path in (PID_ROOT/'source').glob('*'):
        if path.is_file():before[str(path)]=h.sha(path)
    native=PID_ROOT/'build/mycartgs_latency_pid_v1.so'
    before[str(native)]=h.sha(native)
    assert all(h.sha(path)==sha for path,sha in before.items())
    save('PROTECTED_BEFORE.json',before)
    scene,source=p.load_scene(b.prior.b.prior.SLUG)
    student,student_path,payload=p.student_for(source,b.prior.b.prior.SLUG)
    engine=h.engine(source,student,'D')
    size=(int(source.scene.calibration['width']),int(source.scene.calibration['height']))
    assert size==(1200,680)
    camera=h.Camera(source,'D',dm,size)
    binary=b.MODE40/'build/mycartgs_footprint_ninehour_v1.so'
    binary_sha='4de72de81cbfa1eaaee6d065ce643c57ee875be29b34c690891baf389232656e'
    spec=importlib.util.spec_from_file_location('pid_shared_mode40',b.MODE40/'render_runtime.py')
    runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
    runtime.extension=lambda:h.verified('mycartgs_footprint_ninehour_v1',binary,binary_sha)
    raster=runtime.Renderer(40)
    
    modules=new_modules()
    timed_raster=timer_runtime.Renderer(40)
    camera.create(0)
    summary_start=time.perf_counter_ns()
    summary=predictive_runtime.anchor_summary(engine)
    summary_setup_ms=(time.perf_counter_ns()-summary_start)/1e6
    assert int(summary[:,4].sum())<=engine.total
    CONFIGS.update(v2_8192=dict(target_ms=.2,initial_budget=3000,min_budget=256,max_budget=8192),
        v2_expanded=dict(target_ms=.2,initial_budget=3000,min_budget=256,max_budget=engine.total),
        v2_fast=dict(target_ms=.2,initial_budget=3000,min_budget=256,max_budget=engine.total,max_increase=.35,max_decrease=.4),
        predictive=dict(target_ms=.2,initial_budget=3000,min_budget=256,max_budget=engine.total),
        all_candidates=dict(initial_budget=engine.total,min_budget=engine.total,max_budget=engine.total,kp=0.,ki=0.))
    save('SUMMARY_SETUP.json',dict(anchors=len(summary),bytes=summary.nbytes,setup_ms=summary_setup_ms,total=engine.total))
    frames=list(range(160 if args.smoke else 2000))
    repeats=1 if args.smoke else 3
    save('PROTOCOL.json',dict(scene=scene['id'],frames=len(frames),repeats=repeats,width=size[0],height=size[1],
        methods=METHODS,configs=CONFIGS,model_N=43197,renderer='Mode40',network='distilled Compensator + Tensor Core D + new preparation',
        target_ms=.2,band_ms=[.19,.21],paced_pose_groups_hz=90,
        schedule='each pose group interleaves five methods with rotating method order; a new controller instance for each full-trajectory repeat',
        workload='current natural GPU load; no injected work, no stopping or changing external processes; all compared in this interleaved traversal; no historical timing used for claims',
        timing='all methods use the identical native Graph-submit Event pair; one single render per method and frame; no per-frame wait',
        telemetry_poll='after existing engine.prepare synchronization, poll before current timed render; no extra prepare/render, complete prior samples logged outside timing',
        no_oracle=True,no_trial_render=True,quality='separate replay of recorded request budgets after timing; original native RGB without resize; no images saved',
        host_scope='controller request/start/finish API host time recorded separately; telemetry overhead separate; not complete display latency',
        all_requests_retained=True,training=False,production_changed=False))
    save('MODEL_IDENTITIES.json',dict(scene=scene,student=str(student_path),student_sha256=h.sha(student_path),
        controller_binary=str(native),controller_sha256=h.sha(native),mode40_binary=str(binary),mode40_sha256=binary_sha))
    # Bound cold capacity/Graph cost before the recorded traversal; no timing is discarded later.
    timed_raster.precompile(engine.total,camera.create(0),source.scene.background)
    for frame in [0,500,722,1000,1500,1999]:
        cam=camera.create(frame)
        for budget in [256,512,1024,2000,3000,4000,8192,16384,32768,engine.total]:
            packet,ids,stats=engine.prepare(cam,budget=budget)
            raster.render(cam,packet,source.scene.background)
            timed_raster.render(cam,packet,source.scene.background)
    torch.cuda.synchronize()
    rows=[];feedback={};controller_summary=[]
    for repeat in range(repeats):
        controls={method:modules['v3' if method=='predictive' else 'v2'].AsyncPID(CONFIGS[method],0,8) for method in METHODS}
        predictive_runtime.configure(controls['predictive'],summary,camera.factory.projection,*size)
        poses=[np.asarray(source.scene.views[frame]['pose']).ravel().tolist() for frame in frames]
        origin=time.perf_counter()

        def collect(method):
            status=controls[method].stats()
            sample=int(status['last_feedback_frame'])-1
            if sample>=0:
                key=(repeat,method,sample)
                value=dict(render_ms=float(status['last_feedback_ms']),K=int(status['last_feedback_K']))
                if key in feedback:assert feedback[key]==value
                feedback[key]=value
            return status

        for index,frame in enumerate(frames):
            delay=origin+index/90.-time.perf_counter()
            if delay>0:time.sleep(delay)
            actual_group_start=time.perf_counter()
            cam=camera.create(frame)
            for method in p.rotate(METHODS,repeat+index):
                control=controls[method]
                start=time.perf_counter_ns()
                budget=control.request(poses[index],False,-1.)
                request_us=(time.perf_counter_ns()-start)/1000.
                collect(method)
                prep_start=time.perf_counter_ns()
                packet,ids,stats=engine.prepare(cam,budget=budget)
                prepare_host_ms=(time.perf_counter_ns()-prep_start)/1e6
                # Previous render is complete after this required prepare publication fence.
                # At most one outstanding native sample remains; never lose timing snapshots.
                telemetry_start=time.perf_counter_ns()
                control.poll();status=collect(method)
                telemetry_us=(time.perf_counter_ns()-telemetry_start)/1000.
                assert status['pending_samples']==0
                start=time.perf_counter_ns()
                token,begin,end=control.start_native(len(ids),stats['eligible_candidates'])
                start_us=(time.perf_counter_ns()-start)/1000.
                render_start=time.perf_counter_ns()
                image,radii=timed_raster.render(cam,packet,source.scene.background,(begin,end))
                render_host_us=(time.perf_counter_ns()-render_start)/1000.
                start=time.perf_counter_ns();control.finish(token,True)
                finish_us=(time.perf_counter_ns()-start)/1000.
                assert image.shape==(3,size[1],size[0])
                rows.append(dict(repeat=repeat,method=method,frame=frame,frame_start_unix=time.time(),
                    requested_K=budget,K=len(ids),M=stats['candidate_gaussians'],eligible_N=stats['eligible_candidates'],ratio=stats['ratio'],
                    controller_host_us=request_us+start_us+finish_us,telemetry_host_us=telemetry_us,
                    prepare_host_ms=prepare_host_ms,render_host_us=render_host_us,
                    predicted_ms=status.get('predicted_ms',0.),predicted_N=status.get('predicted_available',-1),
                    footprint=status.get('footprint',0.),predicted_pool_limited=status.get('predicted_pool_limited',False),epoch=status['epoch'],integral=status['integral'],
                    view_summary_us=status.get('view_summary_us',0.),summary_host_bytes=status.get('summary_host_bytes',0),
                    group_late_ms=max(0.,(actual_group_start-origin-index/90.)*1000.)))
            if index%200==0:print('TIMING',repeat,index+1,len(frames),controls['predictive'].stats(),flush=True)
        torch.cuda.synchronize()  # Traversal boundary, not a controller feedback wait.
        for method,control in controls.items():
            control.poll();status=collect(method)
            assert status['updates']==len(frames) and status['pending_samples']==0 and status['dropped_samples']==0
            controller_summary.append(dict(repeat=repeat,method=method,**status))
            control.close()
        assert len(feedback)==(repeat+1)*len(frames)*len(METHODS)
        for row in rows:
            measured=feedback[row['repeat'],row['method'],row['frame']]
            assert row['K']==measured['K']
            row['render_ms']=measured['render_ms']
        write('TIMINGS.csv',rows);save('CONTROLLER_SUMMARY.json',controller_summary)
        print('REPEAT_COMPLETE',repeat,flush=True)

    # Each frame/budget is rendered only for quality here; none of these outputs drive PI.
    by_frame={}
    for row in rows:by_frame.setdefault(row['frame'],set()).add(row['requested_K'])
    quality={};quality_rows=[]
    for index,frame in enumerate(frames):
        truth=cv2.imread(source.scene.views[frame]['image'],cv2.IMREAD_COLOR);assert truth is not None
        assert truth.shape[:2]==(size[1],size[0]),(frame,truth.shape,size)
        truth=cv2.cvtColor(truth,cv2.COLOR_BGR2RGB)
        gt=torch.from_numpy(truth).to('cuda').permute(2,0,1)/255.
        cam=camera.create(frame)
        for budget in sorted(by_frame[frame]):
            packet,ids,stats=engine.prepare(cam,budget=budget)
            image,radii=raster.render(cam,packet,source.scene.background)
            psnr=float(-10*torch.log10((image.clamp(0,1)-gt).square().mean()))
            assert np.isfinite(psnr)
            record=dict(frame=frame,requested_K=budget,K=len(ids),M=stats['candidate_gaussians'],ratio=stats['ratio'],psnr=psnr)
            quality[frame,budget]=record;quality_rows.append(record)
        if index%200==0:print('QUALITY',index+1,len(frames),flush=True)
    for row in rows:
        q=quality[row['frame'],row['requested_K']]
        assert row['K']==q['K'] and row['M']==q['M'] and abs(row['ratio']-q['ratio'])<1e-12
        row['psnr']=q['psnr']
    write('REQUESTS_WITH_QUALITY.csv',rows);write('QUALITY_REPLAY.csv',quality_rows)
    after={path:h.sha(path) for path in before};changed=[path for path in before if before[path]!=after[path]]
    save('PROTECTED_AFTER.json',after)
    result=dict(passed=not changed,changed=changed,protected_files=len(before),frames=len(frames),repeats=repeats,
        requests=len(rows),width=size[0],height=size[1],native_resolution=True,GT_resized=False,
        complete_single_request_timings=True,no_samples_dropped=True,natural_GPU_load=True,
        controller_unchanged=False,production_changed=False,training=False,render_images_saved=0)
    save('VALIDATION.json',result);assert result['passed']
    print('COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
