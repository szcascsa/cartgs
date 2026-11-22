"""标定跨帧短周期负载并执行全轨迹速度与PSNR实验。"""
import argparse
import csv
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from reference import (
    ROOT, V27, V27_SHA, CHECKPOINT, VANILLA, DATASET, Monitor,
    Renderer as ReferenceRenderer, BudgetScene, GaussianModel, load_views,
    load_module, save, sha, torch, np,
)
from phase_load import PhaseLoad
from phase_gate import validate_phase_intervals

OFFICE = Path('/home/prmi/zwj/realLOD/saved_experiments/dual_view_li_best_full_20261107/runs/Replica/office3/rgbd/attempt_02/model/41149_shutdown/ply')
OFFICE_CART = Path('/home/prmi/zwj/realLOD/saved_experiments/vanilla_all_datasets_20261031_2011/jobs/replica/rgbd/office3/repeat1/seed0/attempt_2/model/17900_shutdown/ply')
METHODS = ('cartgs_full','ours_full','ours_fixed4000','ours_fixed500','ours_adaptive')
DUTIES = (0.,.4,.6,.8)
BUDGETS = (4000,2000,1000,500)


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def replica_views(checkpoint):
    from scipy.spatial.transform import Rotation
    records=np.loadtxt(checkpoint.parent.parent/'CameraTrajectory_TUM.txt')
    views=[]
    for row in records:
        frame=int(row[0])
        assert row[0]==frame
        pose=np.eye(4)
        pose[:3,:3]=Rotation.from_quat(row[4:8]).as_matrix()
        pose[:3,3]=row[1:4]
        image=Path('/home/prmi/datasets/Replica/office3/results')/f'frame{frame:06}.jpg'
        assert image.is_file()
        views.append(dict(timestamp=float(frame),pose=pose.tolist(),image=str(image),name=image.name))
    assert len(views)==2000
    return views


class Renderer(ReferenceRenderer):
    def __init__(self, scene):
        assert sha(V27)==V27_SHA
        native=load_module('mycartgs_deployment_fp32_v27',V27)
        frozen=load_module('mycartgs_frozen_inference_v3',ROOT/'build/mycartgs_frozen_inference_v3/mycartgs_frozen_inference_v3.so')
        import deployment_runtime
        import frozen_inference_cuda
        deployment_runtime.deployment_extension=lambda:native
        frozen_inference_cuda.extension=lambda:frozen
        checkpoint,vanilla=(CHECKPOINT,VANILLA) if scene=='tum' else (OFFICE,OFFICE_CART)
        self.scene=BudgetScene(checkpoint,DATASET if scene=='tum' else None,4000,0,None,'pytorch')
        if scene=='office3':
            self.scene.views=replica_views(checkpoint)
            cart_views=replica_views(vanilla)
        else:
            calibration,cart_views=load_views(vanilla,DATASET)
            assert calibration==self.scene.calibration
        self.runtime=deployment_runtime.DeploymentRenderer(self.scene.model,self.scene.gaussians,self.scene.config)
        self.cart_pc=GaussianModel(3)
        files=list((vanilla/'point_cloud').glob('iteration_*/point_cloud.ply'))
        assert len(files)==1
        self.cart_pc.load_ply(str(files[0]))
        raster=load_module('original_raster',ROOT/'source/original_raster/__init__.py',True)
        self.cart_renderer=load_module('original_renderer',ROOT/'source/original_renderer.py')
        self.cart_renderer.GaussianRasterizationSettings=raster.GaussianRasterizationSettings
        self.cart_renderer.GaussianRasterizer=raster.GaussianRasterizer
        by_time={view['timestamp']:view for view in cart_views}
        self.cameras=[self.camera(self.scene.calibration,view) for view in self.scene.views]
        self.cart_cameras=[self.camera(self.scene.calibration,by_time[view['timestamp']]) for view in self.scene.views]
        self.library=ctypes.CDLL('/usr/local/cuda-11.8/lib64/libcudart.so')
        least,greatest=ctypes.c_int(),ctypes.c_int()
        assert self.library.cudaDeviceGetStreamPriorityRange(ctypes.byref(least),ctypes.byref(greatest))==0
        self.handle=ctypes.c_void_p()
        self.library.cudaStreamCreateWithPriority.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_uint,ctypes.c_int]
        assert self.library.cudaStreamCreateWithPriority(ctypes.byref(self.handle),1,greatest.value)==0
        self.prepare_stream=torch.cuda.ExternalStream(self.handle.value)
        self.library.cudaStreamGetPriority.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_int)]
        actual=ctypes.c_int()
        assert self.library.cudaStreamGetPriority(self.handle,ctypes.byref(actual))==0
        assert actual.value==greatest.value
        self.render_stream=torch.cuda.default_stream()
        self.priorities=dict(preparation=actual.value,highest=greatest.value,rendering=self.render_stream.priority)
        self.events=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
        torch.cuda.synchronize()
        self.name=scene
        self.metadata=dict(scene=scene,frames=len(self.cameras),gaussians=self.runtime.total,
            cartgs_gaussians=len(self.cart_pc.get_xyz),checkpoint=str(checkpoint),cartgs_checkpoint=str(vanilla),
            priorities=self.priorities,torch=torch.__version__,cuda=torch.version.cuda,v27_sha256=sha(V27))


def budget_for(method,stage):
    if method=='ours_fixed500':
        return 500
    if method=='ours_adaptive':
        return BUDGETS[stage]
    return 4000


def warm(renderer):
    for frame in np.linspace(0,len(renderer.cameras)-1,6,dtype=int):
        for method in METHODS:
            for budget in ((500,1000,2000,4000) if method=='ours_adaptive' else (budget_for(method,0),)):
                renderer.call(int(frame),method,budget)
    started=time.monotonic()
    while time.monotonic()-started<2:
        renderer.call(0,'ours_fixed4000',4000)
        renderer.call(0,'cartgs_full',4000)


def audit(folder,rows,summary,hardware):
    packets=read_csv(folder/'PULSES.csv')
    windows=read_csv(folder/'PULSES.csv.windows.csv')
    assert len(windows)==len(rows)
    interval=validate_phase_intervals(rows,packets)
    assert all(int(p['blocks_each_sm'])==hardware['blocks_per_sm'] for p in packets)
    assert summary['charged_duty']<=summary['target_duty']+1e-10
    assert summary['actual_duty']<=summary['target_duty']+1e-10
    lengths=[int(packet['resident_ns']) for packet in packets]
    maximum=max(lengths,default=0)
    p99=float(np.percentile(lengths,99)) if lengths else 0.
    short_pulse=maximum<=2000000 and p99<=hardware['cycle_ns']*summary['target_duty']+250000
    resident=sum(int(packet['resident_ns']) for packet in packets)
    coverage=sum(int(packet['resident_ns'])*int(packet['unique_sms']) for packet in packets)/(resident*hardware['sm_count']) if resident else 1.
    minimum=min((int(packet['unique_sms'])/hardware['sm_count'] for packet in packets),default=1.)
    interval.update(summary=summary,hardware=hardware,all_pulses_full_sm=minimum==1.,
        sm_coverage_weighted=coverage,sm_coverage_minimum=minimum,
        sm_coverage_passed=coverage>=.99 and minimum>=.95,
        actual_target_error=abs(summary['actual_duty']-summary['target_duty']),
        target_match=abs(summary['actual_duty']-summary['target_duty'])<=.05,
        maximum_on_ns=maximum,p99_on_ns=p99,short_pulse_passed=short_pulse)
    save(folder/'ACCEPTANCE.json',interval)
    return interval


def measure(renderer,folder,method,stage,frames,cycle,repeat,monitor):
    folder.mkdir()
    phase=(.1732050807568877+.3819660112501051*repeat)%1
    duty=DUTIES[stage]
    load=PhaseLoad(ROOT/'build/libphase_load.so',folder/'PULSES.csv',cycle,duty,phase)
    assert load.information['highest_priority']==renderer.priorities['preparation']
    rows=[]
    try:
        for index,frame in enumerate(frames):
            monitor.check()
            row,image=renderer.call(int(frame),method,budget_for(method,stage),duty,load)
            row.update(scene=renderer.name,stage=stage,method=method,repeat=repeat,
                requested_K=budget_for(method,stage) if method not in ('cartgs_full','ours_full') else row['K'],
                calibration_view=int(frame)%17==0,pixel_checked=index in (0,len(frames)-1))
            if row['pixel_checked']:
                signature=hashlib.sha256(image.cpu().numpy().tobytes()).hexdigest()
                _,reference=renderer.call(int(frame),method,budget_for(method,stage))
                assert hashlib.sha256(reference.cpu().numpy().tobytes()).hexdigest()==signature
            rows.append(row)
            if index%200==0:
                save(ROOT/'reports/ACTIVE_CASE.json',dict(scene=renderer.name,method=method,stage=stage,
                     repeat=repeat,frame=int(frame)+1,done=index+1,total=len(frames),unix=time.time()))
                print('PROGRESS',renderer.name,method,stage,repeat,index,len(frames),flush=True)
        summary=load.finish()
    finally:
        load.close()
        if rows:
            write_csv(folder/'TIMINGS.csv',rows)
    result=audit(folder,rows,summary,load.information)
    save(folder/'SUMMARY.json',summary)
    return rows,result


def calibrate(folder,monitor):
    locked=dict(duties=DUTIES,budgets=BUDGETS,latency_factor=1.25,near_full_psnr_drop_db=1.,
                light_vs_fixed500_gain_db=1.,duty_tolerance=.05,calibration_duty_tolerance=.03,
                maximum_on_ns=2000000,scenes={})
    target_path=ROOT/'reports/LATENCY_TARGETS_LOCK.json'
    if target_path.exists():
        locked['scenes']=json.loads(target_path.read_text())
    for scene in ('tum','office3'):
        if scene in locked['scenes']:
            continue
        with torch.inference_mode(False),torch.no_grad():
            renderer=Renderer(scene)
        try:
            with torch.inference_mode():
                warm(renderer)
                frames=list(range(0,len(renderer.cameras),17))
                times=[]
                for repeat in range(3):
                    for frame in frames:
                        row,_=renderer.call(frame,'ours_fixed4000',4000)
                        row.update(repeat=repeat)
                        times.append(row)
                write_csv(folder/f'{scene}_CALIBRATION_BASELINE.csv',times)
                tau=1.25*float(np.percentile([row['request_ms'] for row in times],95))
                locked['scenes'][scene]=dict(tau_ms=tau,metadata=renderer.metadata,calibration_frames=frames)
        finally:
            renderer.close()
            del renderer
            torch.cuda.empty_cache()
    chosen=None
    candidate_results=[]
    for cycle in (500000,1000000,2000000):
        successes=[]
        for scene in ('tum','office3'):
            with torch.inference_mode(False),torch.no_grad():
                renderer=Renderer(scene)
            try:
                with torch.inference_mode():
                    warm(renderer)
                    frames=np.linspace(0,len(renderer.cameras)-1,96,dtype=int).tolist()
                    for stage in (1,2,3):
                        for method in ('cartgs_full','ours_fixed500'):
                            target=folder/f'cycle_{cycle}_{scene}_{method}_{stage}'
                            _,result=measure(renderer,target,method,stage,frames,cycle,0,monitor)
                            successes.append(result['actual_target_error']<=.03 and result['short_pulse_passed'] and result['sm_coverage_passed'])
                            print('CALIBRATION',cycle,scene,method,DUTIES[stage],result['summary']['actual_duty'],flush=True)
            finally:
                renderer.close()
                del renderer
                torch.cuda.empty_cache()
        candidate_results.append(dict(cycle_ns=cycle,passed=all(successes),conditions=len(successes)))
        save(folder/'CYCLE_CANDIDATES.json',candidate_results)
        if all(successes):
            chosen=cycle
            break
    assert chosen is not None, 'No short-cycle candidate matched actual duty'
    locked['cycle_ns']=chosen
    save(ROOT/'reports/PROTOCOL_LOCK.json',locked)
    print('PROTOCOL_LOCKED',chosen,{s:v['tau_ms'] for s,v in locked['scenes'].items()},flush=True)


def benchmark(folder,monitor):
    locked=json.loads((ROOT/'reports/PROTOCOL_LOCK.json').read_text())
    all_rows=[]
    for scene in ('tum','office3'):
        with torch.inference_mode(False),torch.no_grad():
            renderer=Renderer(scene)
        try:
            with torch.inference_mode():
                warm(renderer)
                frames=list(range(len(renderer.cameras)))
                for repeat in range(3):
                    order=list(range(4))
                    order=order[repeat:]+order[:repeat]
                    for stage in order:
                        rotation=(repeat+stage)%len(METHODS)
                        methods=METHODS[rotation:]+METHODS[:rotation]
                        for method in methods:
                            target=folder/f'{scene}_{method}_{stage}_{repeat}'
                            rows,result=measure(renderer,target,method,stage,frames,locked['cycle_ns'],repeat,monitor)
                            assert result['target_match'] and result['short_pulse_passed'] and result['sm_coverage_passed'], ('Load validation failed',target,result)
                            all_rows.extend(rows)
        finally:
            renderer.close()
            del renderer
            torch.cuda.empty_cache()
    write_csv(folder/'ALL_TIMINGS.csv',all_rows)


def quality(folder,monitor):
    rows=[]
    for scene in ('tum','office3'):
        with torch.inference_mode(False),torch.no_grad():
            renderer=Renderer(scene)
        try:
            with torch.inference_mode():
                warm(renderer)
                for frame in range(len(renderer.cameras)):
                    monitor.check()
                    reference=renderer.scene.ground_truth(frame)
                    requests=[('cartgs_full',0),('ours_full',0)]+[('ours_adaptive',stage) for stage in range(4)]
                    for method,stage in requests:
                        record,image=renderer.call(frame,method,budget_for(method,stage))
                        psnr=float(-10*torch.log10((image.clamp(0,1)-reference).square().mean()))
                        rows.append(dict(scene=scene,frame=frame+1,method=method,budget=record['K'],
                            requested_budget=budget_for(method,stage),radii_positive=record['radii_positive'],psnr=psnr,
                            calibration_view=frame%17==0))
                    if frame%200==0:
                        print('QUALITY',scene,frame,len(renderer.cameras),flush=True)
        finally:
            renderer.close()
            del renderer
            torch.cuda.empty_cache()
    write_csv(folder/'QUALITY.csv',rows)


def curves(folder,monitor):
    locked=json.loads((ROOT/'reports/PROTOCOL_LOCK.json').read_text())
    rows=[]
    schedule=(0,1,2,3,3,2,1,0)
    for scene in ('tum','office3'):
        with torch.inference_mode(False),torch.no_grad():
            renderer=Renderer(scene)
        try:
            with torch.inference_mode():
                warm(renderer)
                chunks=np.array_split(np.arange(len(renderer.cameras)),len(schedule))
                for method in METHODS:
                    for segment,(stage,frames) in enumerate(zip(schedule,chunks)):
                        target=folder/f'{scene}_{method}_segment{segment}'
                        measured,result=measure(renderer,target,method,stage,frames.tolist(),locked['cycle_ns'],0,monitor)
                        for row in measured:
                            row['segment']=segment
                        rows.extend(measured)
        finally:
            renderer.close()
            del renderer
            torch.cuda.empty_cache()
    write_csv(folder/'CURVE_TIMINGS.csv',rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['calibrate','benchmark','quality','curves'])
    parser.add_argument('--label',required=True)
    args=parser.parse_args()
    lock=(ROOT/'reports/run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    folder=ROOT/'runs'/args.label
    folder.mkdir()
    protected=json.loads((ROOT/'reports/PROTECTED_BEFORE.json').read_text())
    assert all(sha(Path(name))==value for name,value in protected.items())
    monitor=Monitor(folder)
    state=dict(state='running',stage=args.stage,label=args.label,pid=os.getpid(),
        start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    save(ROOT/'reports/STATUS.json',state)
    try:
        globals()[args.stage](folder,monitor)
        assert all(sha(Path(name))==value for name,value in protected.items())
        monitor.close()
        state.update(state='complete',protected_files=len(protected))
    except Exception as error:
        state.update(state='failed',error=repr(error))
        raise
    finally:
        if monitor.thread.is_alive():
            monitor.close()
        save(ROOT/'reports/STATUS.json',state)


if __name__=='__main__':
    main()
