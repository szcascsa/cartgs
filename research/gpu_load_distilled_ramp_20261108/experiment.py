"""蒸馏学生、逐帧线性负载与预算的独立部署测速。"""
import argparse
import csv
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from reference import (
    ROOT, BudgetScene, GaussianModel, Renderer as ReferenceRenderer,
    load_module, save, sha, torch, np, gpu_state,
)
from previous_experiment import OFFICE, OFFICE_CART, replica_views
from phase_load import PhaseLoad
from phase_gate import validate_phase_intervals
from run_guard import implementation_identity, require_gpu_acceptance, write_gpu_acceptance

STUDENT=Path('/home/prmi/zwj/experiments/compensator_distillation_replica_all_20261108/models/Replica__office3__rgbd/continuous_calibration/best.pt')
STUDENT_SHA='20bcf17294a4eeec7d493246742882d04767a74e7ddfd3dff50fd4349fb6ac90'
BINARY=ROOT/'build/mycartgs_deployment_fp32_v31/mycartgs_deployment_fp32_v31.so'
BINARY_SHA='fa7b855685ecd4f75ce7118ee07fadd27640af08099a6775ffee5fe87fa7d601'
METHODS=('cartgs_full','ours_full','student_fixed4000','student_fixed1000','student_adaptive')
CYCLE_NS=2000000


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def schedule(frame):
    progress=frame/1999
    return .2+.6*progress,int(np.floor(4000-3000*progress+.5))


class Monitor:
    def __init__(self,folder):
        self.folder=folder
        self.resident=json.loads((ROOT/'reports/RESIDENT_PROCESS.json').read_text())
        self.error=None
        self.stop=threading.Event()
        self.ready=threading.Event()
        self.thread=threading.Thread(target=self.run,daemon=True)
        self.thread.start()
        assert self.ready.wait(15), 'GPU monitor did not start'
        self.check()

    def run(self):
        try:
            with (self.folder/'GPU_MONITOR.jsonl').open('w',buffering=1) as stream:
                while not self.stop.is_set():
                    record=gpu_state()
                    for process in record['foreign']:
                        if process['pid']!=self.resident['pid']:
                            raise RuntimeError('New CUDA process '+str(process))
                        identity=Path(f"/proc/{process['pid']}/stat").read_text().split()[21]
                        assert identity==self.resident['start_ticks']
                    raw=subprocess.check_output(['nvidia-smi','pmon','-c','1','-s','um'],text=True)
                    for line in raw.splitlines():
                        fields=line.split()
                        if not fields or fields[0].startswith('#') or len(fields)<7:
                            continue
                        if fields[2]=='C' and int(fields[1])!=os.getpid():
                            assert int(fields[1])==self.resident['pid'],line
                            assert all(value=='-' or int(value)==0 for value in fields[3:7]),line
                    record['pmon']=raw
                    record['permitted_idle_resident']=self.resident['pid']
                    stream.write(json.dumps(record)+'\n')
                    self.ready.set()
                    self.stop.wait(1.)
        except Exception as error:
            self.error=repr(error)
            self.ready.set()

    def check(self):
        if self.error:
            raise RuntimeError(self.error)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)
        assert not self.thread.is_alive()
        self.check()


class Renderer(ReferenceRenderer):
    def __init__(self):
        assert sha(BINARY)==BINARY_SHA
        assert sha(STUDENT)==STUDENT_SHA
        native=load_module('mycartgs_deployment_fp32_v31',BINARY)
        frozen=load_module('mycartgs_frozen_inference_v3',ROOT/'build/mycartgs_frozen_inference_v3/mycartgs_frozen_inference_v3.so')
        import deployment_runtime
        import frozen_inference_cuda
        from compensation_student import load_student
        deployment_runtime.deployment_extension=lambda:native
        frozen_inference_cuda.extension=lambda:frozen
        self.scene=BudgetScene(OFFICE,None,4000,0,None,'pytorch')
        self.scene.views=replica_views(OFFICE)
        cart_views=replica_views(OFFICE_CART)
        student,payload=load_student(STUDENT,'cuda')
        self.runtime=deployment_runtime.DeploymentRenderer(self.scene.model,self.scene.gaussians,self.scene.config,student=student)
        self.cart_pc=GaussianModel(3)
        paths=list((OFFICE_CART/'point_cloud').glob('iteration_*/point_cloud.ply'))
        assert len(paths)==1
        self.cart_pc.load_ply(str(paths[0]))
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
        self.events=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
        self.priorities=dict(preparation=actual.value,highest=greatest.value,rendering=self.render_stream.priority)
        self.metadata=dict(scene='Replica/office3/rgbd',views=2000,gaussians=self.runtime.total,
            cartgs_gaussians=len(self.cart_pc.get_xyz),student=str(STUDENT),student_sha256=sha(STUDENT),
            student_parameters=sum(p.numel() for p in student.parameters()),teacher_fingerprint=payload['teacher_fingerprint'],
            checkpoint=str(OFFICE),cartgs_checkpoint=str(OFFICE_CART),binary_sha256=sha(BINARY),backend='v31',
            priorities=self.priorities,torch=torch.__version__,cuda=torch.version.cuda)
        assert self.metadata['student_parameters']==3534
        torch.cuda.synchronize()

    def call(self,frame,method,budget,duty=0.,load=None):
        row,image=super().call(frame,method,budget,duty,load)
        row['requested_K']=row['K'] if method in ('cartgs_full','ours_full') else budget
        return row,image


def budget_for(method,frame):
    if method=='student_fixed1000':
        return 1000
    if method=='student_adaptive':
        return schedule(frame)[1]
    return 4000


def warm(renderer):
    for frame in np.linspace(0,1999,7,dtype=int):
        for budget in (1000,1024,1025,2048,2049,3000,4000):
            renderer.call(int(frame),'student_adaptive',budget)
        renderer.call(int(frame),'ours_full',4000)
        renderer.call(int(frame),'cartgs_full',4000)
    start=time.monotonic()
    while time.monotonic()-start<2:
        renderer.call(1000,'student_adaptive',2534)
        renderer.call(1000,'cartgs_full',4000)


def audit(folder,rows,summary,hardware,strict_bins):
    packets=read_csv(folder/'PULSES.csv')
    windows=read_csv(folder/'PULSES.csv.windows.csv')
    assert len(windows)==len(rows)
    interval=validate_phase_intervals(rows,packets)
    assert summary['charged_duty']<=summary['weighted_target_duty']+1e-10
    assert summary['charged_duty']<=.8+1e-10
    assert all(0<=float(row['target_duty'])<=.8 for row in windows)
    on=[int(packet['resident_ns']) for packet in packets]
    assert max(on,default=0)<=2000000
    resident=sum(on)
    coverage=sum(int(p['resident_ns'])*int(p['unique_sms']) for p in packets)/(resident*hardware['sm_count']) if resident else 1.
    minimum=min((int(p['unique_sms'])/hardware['sm_count'] for p in packets),default=1.)
    sm_area=sum(float(p['sm_equivalent_ns']) for p in packets)
    slot_area=sum(float(p['slot_equivalent_ns']) for p in packets)
    charged=sum(int(p['charged_ns']) for p in packets)
    assert charged==summary['charged_ns']==sum(int(w['charged_ns']) for w in windows)
    active=sum(int(w['active_ns']) for w in windows)
    assert active==summary['active_ns']
    assert abs(sm_area-summary['sm_equivalent_ns'])<=1.
    assert all(-1e-6<=int(p['charged_ns'])-float(p['sm_equivalent_ns'])<1.000001 for p in packets)
    settled=active+summary['tail_ns']
    assert abs(sm_area/settled-summary['actual_sm_duty'])<=1e-10
    assert sm_area/settled<=.8+1e-10
    windows_by_id={row['generation']:row for row in windows}
    bins=[]
    for index in range(8):
        selected=[row for row in rows if (int(row['frame'])-1)//250==index]
        if not selected:
            continue
        grouped=[windows_by_id[str(row['load_generation'])] for row in selected]
        elapsed=sum(int(w['active_ns']) for w in grouped)
        actual=sum(int(w['resident_ns']) for w in grouped)/elapsed
        charged=sum(int(w['charged_ns']) for w in grouped)/elapsed
        target=sum(float(w['target_duty'])*int(w['active_ns']) for w in grouped)/elapsed
        generations={w['generation'] for w in grouped}
        pieces=[p for p in packets if p['generation'] in generations]
        sm_actual=sum(float(p['sm_equivalent_ns']) for p in pieces)/elapsed
        slot_actual=sum(float(p['slot_equivalent_ns']) for p in pieces)/elapsed
        bins.append(dict(bin=index,samples=len(selected),resident_envelope_duty=actual,charged_duty=charged,
                        weighted_target=target,error=actual-target,sm_actual_duty=sm_actual,
                        slot_actual_duty=slot_actual,sm_error=sm_actual-target))
    interval.update(summary=summary,hardware=hardware,bins=bins,maximum_on_ns=max(on,default=0),
        p99_on_ns=float(np.percentile(on,99)) if on else 0.,sm_coverage_weighted=coverage,
        sm_coverage_minimum=minimum,sm_area_coverage=sm_area/resident,slot_area_coverage=slot_area/resident,
        load_packets=len(packets),
        bin_match=all(abs(row['sm_error'])<=.05 for row in bins))
    save(folder/'ACCEPTANCE.json',interval)
    assert coverage>=.99, coverage
    assert hardware['billing_sm_area']
    if strict_bins:
        assert interval['bin_match'],bins


def preflight(renderer,folder,monitor):
    checks=[]
    with torch.inference_mode():
        for frame in (0,249,999,1499,1999):
            for budget in (1000,1024,1025,2000,2048,2049,2534,4000):
                cam=renderer.cameras[frame]
                with torch.cuda.stream(renderer.prepare_stream):
                    packet,ids,_=renderer.runtime.prepare(cam,budget=budget)
                renderer.prepare_stream.synchronize()
                renderer.runtime._pipeline._graph_enabled(False)
                with torch.cuda.stream(renderer.prepare_stream):
                    native,native_ids,_=renderer.runtime.prepare(cam,budget=budget)
                renderer.prepare_stream.synchronize()
                renderer.runtime._pipeline._graph_enabled(True)
                assert torch.equal(ids,native_ids)
                for name,value in packet['inputs'].items():
                    if value is not None:
                        assert torch.equal(value.contiguous().view(torch.uint8),native['inputs'][name].contiguous().view(torch.uint8)),(frame,budget,name)
                checks.append(dict(frame=frame+1,budget=budget,K=len(ids),bitwise=True))
            monitor.check()
        save(folder/'GRAPH_NATIVE_PARITY.json',dict(passed=True,cases=checks))
        warm(renderer)
        frames=list(range(2000))
        for method in ('cartgs_full','ours_full','student_fixed1000'):
            measure(renderer,folder/method,method,frames,0,monitor,strict_bins=True,quality=False)


def measure(renderer,folder,method,frames,repeat,monitor,strict_bins=True,quality=True):
    folder.mkdir()
    phase=(.1732050807568877+.3819660112501051*repeat)%1
    load=PhaseLoad(ROOT/'build/libramp_load.so',folder/'PULSES.csv',CYCLE_NS,schedule(frames[0])[0],phase)
    rows=[]
    try:
        for index,frame in enumerate(frames):
            monitor.check()
            duty,_=schedule(frame)
            budget=budget_for(method,frame)
            row,image=renderer.call(frame,method,budget,duty,load)
            row.update(method=method,repeat=repeat,load_target=duty,adaptive_budget=schedule(frame)[1],
                       source_student=method not in ('cartgs_full','ours_full'),pixel_checked=index in (0,len(frames)-1))
            if row['pixel_checked']:
                signature=hashlib.sha256(image.cpu().numpy().tobytes()).hexdigest()
                _,reference=renderer.call(frame,method,budget)
                assert hashlib.sha256(reference.cpu().numpy().tobytes()).hexdigest()==signature
            if quality:
                truth=renderer.scene.ground_truth(frame)
                row['psnr']=float(-10*torch.log10((image.clamp(0,1)-truth).square().mean()))
            rows.append(row)
            if index%250==0:
                save(ROOT/'reports/ACTIVE_CASE.json',dict(method=method,repeat=repeat,frame=frame+1,
                    completed=index+1,total=len(frames),target_duty=duty,budget=budget,unix=time.time()))
                print('PROGRESS',method,repeat,frame+1,len(frames),flush=True)
        summary=load.finish()
    finally:
        load.close()
        if rows:
            write_csv(folder/'TIMINGS.csv',rows)
    save(folder/'SUMMARY.json',summary)
    audit(folder,rows,summary,load.information,strict_bins)
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--label',required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    lock=(ROOT/'reports/run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity=implementation_identity(ROOT)
    accepted=None
    if not args.smoke:
        accepted=require_gpu_acceptance(ROOT,identity)
    folder=ROOT/'runs'/args.label
    folder.mkdir()
    save(folder/'IMPLEMENTATION.json',identity)
    if accepted:
        save(folder/'GPU_ACCEPTANCE_USED.json',accepted)
    protected=json.loads((ROOT/'reports/PROTECTED_BEFORE.json').read_text())
    assert all(sha(Path(path))==value for path,value in protected.items())
    monitor=Monitor(folder)
    renderer=None
    status=dict(state='starting',label=args.label,pid=os.getpid(),
        start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    save(ROOT/'reports/STATUS.json',status)
    try:
        with torch.inference_mode(False),torch.no_grad():
            renderer=Renderer()
        save(folder/'ENVIRONMENT.json',renderer.metadata)
        if args.smoke:
            preflight(renderer,folder,monitor)
        else:
            records=[]
            with torch.inference_mode():
                warm(renderer)
                for repeat in range(3):
                    order=METHODS[repeat:]+METHODS[:repeat]
                    for method in order:
                        records.extend(measure(renderer,folder/f'{repeat}_{method}',method,list(range(2000)),repeat,monitor))
                        status.update(state='timing',repeat=repeat,method=method,rows=len(records))
                        save(ROOT/'reports/STATUS.json',status)
            write_csv(folder/'ALL_TIMINGS.csv',records)
            assert len(records)==30000
        assert all(sha(Path(path))==value for path,value in protected.items())
        status.update(state='complete',protected_files=len(protected))
    except Exception as error:
        status.update(state='failed',error=repr(error))
        raise
    finally:
        try:
            if renderer:
                renderer.close()
            monitor.close()
        except Exception as error:
            status.update(state='failed',cleanup_error=repr(error))
            raise
        finally:
            save(ROOT/'reports/STATUS.json',status)
    if args.smoke:
        try:
            write_gpu_acceptance(ROOT,folder,identity)
            status['gpu_acceptance_ready']=True
        except Exception as error:
            status.update(state='failed',error=repr(error))
            raise
        finally:
            save(ROOT/'reports/STATUS.json',status)


if __name__=='__main__':
    main()
