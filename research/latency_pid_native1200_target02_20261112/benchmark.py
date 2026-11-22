"""原始1200x680分辨率：0.2ms目标、原生PI、完整单次请求计时。"""
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
from latency_pid import pid_extension
PRIOR=ROOT.parent/'office2_dynamic04_mode40_20261112'
spec=importlib.util.spec_from_file_location('pid_shared_prior',PRIOR/'benchmark.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
h,p,b,torch,np=d.h,d.p,d.b,d.torch,d.np
import cv2

METHODS=['fixed_3000','fixed_256','pid_02']
CONFIGS={'fixed_3000':dict(initial_budget=3000,min_budget=3000,max_budget=3000,kp=0.,ki=0.),
         'fixed_256':dict(initial_budget=256,min_budget=256,max_budget=256,kp=0.,ki=0.),
         'pid_02':dict(target_ms=.2,initial_budget=3000,min_budget=256,max_budget=8192,kp=1.,ki=4.,kd=0.)}


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
    extension=pid_extension(str(PID_ROOT/'build'))
    frames=list(range(80 if args.smoke else 2000))
    repeats=1 if args.smoke else 3
    save('PROTOCOL.json',dict(scene=scene['id'],frames=len(frames),repeats=repeats,width=size[0],height=size[1],
        methods=METHODS,configs=CONFIGS,model_N=43197,renderer='Mode40',network='distilled Compensator + Tensor Core D + new preparation',
        target_ms=.2,band_ms=[.19,.21],paced_pose_groups_hz=90,
        schedule='each pose group interleaves three methods with rotating method order; a new PI instance for each full-trajectory repeat',
        workload='current natural GPU load; no injected work, no stopping or changing external processes; compare previous 0.4ms run with load caveat',
        timing='one native CUDA Event pair per request, original AsyncPID Event ring; every completed sample logged once; no per-frame synchronization added',
        telemetry_poll='after existing engine.prepare synchronization, poll before current timed render; no extra prepare/render, complete prior samples logged outside timing',
        no_oracle=True,no_trial_render=True,quality='separate replay of recorded request budgets after timing; original native RGB without resize; no images saved',
        host_scope='controller request/start/finish API host time recorded separately; telemetry overhead separate; not complete display latency',
        all_requests_retained=True,training=False,production_changed=False))
    save('MODEL_IDENTITIES.json',dict(scene=scene,student=str(student_path),student_sha256=h.sha(student_path),
        controller_binary=str(native),controller_sha256=h.sha(native),mode40_binary=str(binary),mode40_sha256=binary_sha))
    # Bound cold capacity/Graph cost before the recorded traversal; no timing is discarded later.
    for frame in [0,500,722,1000,1500,1999]:
        cam=camera.create(frame)
        for budget in [256,512,1024,2000,3000,4000,8192]:
            packet,ids,stats=engine.prepare(cam,budget=budget)
            raster.render(cam,packet,source.scene.background)
    torch.cuda.synchronize()
    rows=[];feedback={};controller_summary=[]
    for repeat in range(repeats):
        controls={method:extension.AsyncPID(CONFIGS[method],0,8) for method in METHODS}
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
                token=control.start(len(ids),stats['eligible_candidates'])
                start_us=(time.perf_counter_ns()-start)/1000.
                image,radii=raster.render(cam,packet,source.scene.background)
                assert image.shape==(3,size[1],size[0])
                start=time.perf_counter_ns()
                control.finish(token,True)
                finish_us=(time.perf_counter_ns()-start)/1000.
                rows.append(dict(repeat=repeat,method=method,frame=frame,frame_start_unix=time.time(),
                    requested_K=budget,K=len(ids),M=stats['candidate_gaussians'],eligible_N=stats['eligible_candidates'],ratio=stats['ratio'],
                    controller_host_us=request_us+start_us+finish_us,telemetry_host_us=telemetry_us,
                    prepare_host_ms=prepare_host_ms,epoch=status['epoch'],integral=status['integral'],
                    group_late_ms=max(0.,(actual_group_start-origin-index/90.)*1000.)))
            if index%200==0:print('TIMING',repeat,index+1,len(frames),controls['pid_02'].stats(),flush=True)
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
        controller_unchanged=True,production_changed=False,training=False,render_images_saved=0)
    save('VALIDATION.json',result);assert result['passed']
    print('COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
