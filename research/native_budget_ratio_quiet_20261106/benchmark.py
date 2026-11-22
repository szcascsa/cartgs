"""补充ratio质量；以独立进程活动决定整轮计时是否可用。"""
from common import *
import fcntl
import importlib.util
import math
import random
import threading
import shutil
import numpy as np
import torch
from budget_viewer import BudgetScene
import deployment_runtime
import frozen_inference_cuda
from renderer import render_prepared
from utils import MiniCam,focal2fov

PREVIOUS=Path('/home/prmi/zwj/experiments/native_budget100_500_fr1xyz_20261106')
REQUESTS=[dict(method=f'ours_{b}',budget=b) for b in (100,200,300,400,500)]+[
    dict(method='ratio_50',ratio_budget=.5),dict(method='ratio_80',ratio_budget=.8)]


def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def load_binary(name):
    path=ROOT/'build'/name/(name+'.so')
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def bitwise(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))


def counted(binding,camera,data,bg):
    inputs=data['inputs'];empty=torch.Tensor([]);value=lambda k:inputs[k] if inputs[k] is not None else empty
    return binding.render_with_counts(bg,value('means3D'),value('colors_precomp'),value('opacities'),value('scales'),value('rotations'),1.,
        value('cov3D_precomp'),camera.world_view_transform,camera.full_proj_transform,math.tan(camera.FoVx*.5),math.tan(camera.FoVy*.5),
        camera.image_height,camera.image_width,value('dc'),value('shs'),data['sh_degree'],camera.camera_center,False,False,True,binding.Mode.Inference)


class ActivityMonitor:
    def __init__(self):
        self.samples=[];self.error=None;self.own_pid=os.getpid()
        self.process=subprocess.Popen(['nvidia-smi','pmon','-s','um','-o','T','-d','1'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        self.thread=threading.Thread(target=self.collect,daemon=True);self.thread.start()

    def collect(self):
        current=None;records=[];arrival=None
        try:
            with (ROOT/'reports/PMON_RAW.log').open('w',buffering=1) as log:
                for line in self.process.stdout:
                    log.write(line);fields=line.split()
                    if not fields or fields[0].startswith('#'):continue
                    assert len(fields)>=9,line
                    if fields[0]!=current:
                        if current is not None:
                            self.samples.append(dict(clock=current,receipt_unix=arrival,records=records,
                                external_active=any(v['pid']!=self.own_pid and any(v[k]>0 for k in ('sm','mem','enc','dec')) for v in records)))
                        current=fields[0];records=[];arrival=time.time()
                    number=lambda x:0 if x=='-' else int(x)
                    records.append(dict(gpu=int(fields[1]),pid=number(fields[2]),kind=fields[3],
                        sm=number(fields[4]),mem=number(fields[5]),enc=number(fields[6]),dec=number(fields[7]),command=fields[-1]))
        except Exception as e:self.error=repr(e)

    def idle(self):
        assert self.error is None,self.error
        assert self.process.poll() is None,'pmon exited'
        samples=list(self.samples)
        return len(samples)>=2 and time.time()-samples[-1]['receipt_unix']<3 and not any(s['external_active'] for s in samples[-2:])

    def assess(self,start,end):
        assert self.error is None,self.error
        samples=[s for s in list(self.samples) if start-2<=s['receipt_unix']<=end+3]
        covered=len(samples)>=math.floor(end-start)+4 and samples[0]['receipt_unix']<=start-1 and samples[-1]['receipt_unix']>=end+2
        return dict(accepted=covered and not any(s['external_active'] for s in samples),coverage=covered,samples=samples)

    def close(self):
        self.process.terminate();self.thread.join(timeout=3)
        save(ROOT/'reports/PMON_SAMPLES.json',self.samples)


@torch.no_grad()
def main():
    lock=(ROOT/'reports/run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'STATUS.json').exists()
    state=dict(state='loading',process=identity(),started_unix=time.time());save(ROOT/'STATUS.json',state)
    torch.set_num_threads(2);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    protected=json.loads((ROOT/'reports/INPUTS_BEFORE.json').read_text());assert all(sha(n)==h for n,h in protected.items())
    source=json.loads((ROOT/'reports/RUNTIME_SOURCE_HASHES.json').read_text());assert all(sha(ROOT/'source'/n)==h for n,h in source.items())
    # 只替换扩展加载器，加载上轮已验证的相同二进制副本；prepare／MLP实现不变。
    deployment_runtime.deployment_extension=lambda:load_binary('mycartgs_deployment_fp32_v4')
    frozen_inference_cuda.extension=lambda:load_binary('mycartgs_frozen_inference_v3')
    scene=BudgetScene(CHECKPOINT,DATASET,100,0,None,'pytorch');scene.model.eval()
    runtime=deployment_runtime.DeploymentRenderer.from_checkpoint(CHECKPOINT,0);assert runtime.total==13298 and len(scene.views)==792
    c=scene.calibration;cameras=[MiniCam(c['width'],c['height'],focal2fov(c['fx'],c['width']),focal2fov(c['fy'],c['height']),np.linalg.inv(np.asarray(v['pose']))) for v in scene.views]
    path=Path('/home/prmi/zwj/experiments/fr1xyz_viewer_budget_comparison_20261105/build/fr1xyz_production_counts.so')
    spec=importlib.util.spec_from_file_location('fr1xyz_production_counts',path);binding=importlib.util.module_from_spec(spec);spec.loader.exec_module(binding)
    save(ROOT/'reports/ENVIRONMENT.json',dict(checkpoint=str(CHECKPOINT),N=runtime.total,anchors=2790,frames=792,requests=REQUESTS,
        backend=runtime.backend,binary=str(runtime._ext.__file__),binary_sha256=sha(runtime._ext.__file__),torch=torch.__version__,cuda=torch.version.cuda,
        tf32=False,device=torch.cuda.get_device_name(),quality_reused_fixed_budgets=True,cartgs_history=True,images_saved=0,training_runs=0))
    def prepare(frame,request):
        args={k:v for k,v in request.items() if k!='method'};data,ids,stats=runtime.prepare(cameras[frame],**args)
        target=request.get('budget',math.floor(request.get('ratio_budget',0)*stats['candidate_gaussians']))
        assert len(ids)==min(target,stats['eligible_candidates']) and stats['requested_budget']==target and not stats['cache_hit']
        return data,ids,stats
    def call(frame,request):
        torch.cuda.synchronize();a=time.perf_counter();data,ids,stats=prepare(frame,request);b=time.perf_counter()
        render_prepared(cameras[frame],data,scene.background);torch.cuda.synchronize();end=time.perf_counter()
        return dict(method=request['method'],frame=frame,K=len(ids),candidate_N=stats['candidate_gaussians'],eligible_N=stats['eligible_candidates'],
            network_ratio=stats['ratio'],requested_K=stats['requested_budget'],shortfall=stats['budget_shortfall'],prepare_ms=(b-a)*1000,render_ms=(end-b)*1000,total_ms=(end-a)*1000)
    with torch.inference_mode():
        parity=[]
        for frame in (0,396,791):
            for req in REQUESTS:
                data,ids,stats=prepare(frame,req);normal=render_prepared(cameras[frame],data,scene.background,return_radii=True);output=counted(binding,cameras[frame],data,scene.background)
                assert bitwise(normal['render'],output[2]) and torch.equal(normal['radii'],output[3])
                parity.append(dict(frame=frame,method=req['method'],render_and_radii_bitwise=True))
        save(ROOT/'reports/COUNT_RENDERER_PARITY.json',dict(passed=True,cases=parity))
        rows=[]
        for frame,v in enumerate(scene.views):
            gt=scene.ground_truth(frame)
            for req in REQUESTS[-2:]:
                data,ids,stats=prepare(frame,req);output=counted(binding,cameras[frame],data,scene.background);image=output[2].clamp(0,1)
                psnr=float(-10*torch.log10((image-gt).square().mean()));assert math.isfinite(psnr) and torch.isfinite(output[2]).all()
                row=dict(method=req['method'],frame=frame,timestamp=v['timestamp'],image=v['image'],N=13298,K=len(ids),candidate_N=stats['candidate_gaussians'],eligible_N=stats['eligible_candidates'],
                    network_ratio=stats['ratio'],shortfall=stats['budget_shortfall'],requested_K=stats['requested_budget'],
                    projected_K=int((output[3]>0).sum()),contributing_K=int((output[5]>0).sum()),psnr=psnr,photo_l1=float((image-gt).abs().mean()))
                assert 0<=row['contributing_K']<=row['projected_K']<=row['K'];rows.append(row)
            if frame%100==0 or frame==791:
                state.update(state='quality',frames=frame+1,quality_rows=len(rows));save(ROOT/'STATUS.json',state);print(json.dumps(state),flush=True)
        write(ROOT/'evaluation/RATIO_PER_VIEW.csv',rows);assert len(rows)==1584
        for frame in np.linspace(0,791,64,dtype=int):
            for req in REQUESTS:call(int(frame),req)
        observations=[gpu('timing_wait_start')];monitor=ActivityMonitor();deadline=time.time()+900;accepted=[];attempts=[];all_rows=[]
        try:
            while len(accepted)<3 and time.time()<deadline:
                if not monitor.idle():
                    state.update(state='waiting_quiet_gpu',accepted_rounds=len(accepted),remaining_seconds=round(deadline-time.time()))
                    save(ROOT/'STATUS.json',state);time.sleep(1);continue
                repeat=len(accepted);frames=list(range(792))
                if repeat==1:frames.reverse()
                if repeat==2:random.Random(0).shuffle(frames)
                attempt=len(attempts);start=time.time();values=[]
                state.update(state='timing',repeat=repeat,attempt=attempt);save(ROOT/'STATUS.json',state)
                for index,frame in enumerate(frames):
                    shift=(index+repeat)%len(REQUESTS)
                    for req in REQUESTS[shift:]+REQUESTS[:shift]:
                        value=call(frame,req);value.update(repeat=repeat,order=index,attempt=attempt);values.append(value)
                end=time.time();time.sleep(4);decision=monitor.assess(start,end)
                info=dict(attempt=attempt,repeat=repeat,start_unix=start,end_unix=end,rows=len(values),**decision);attempts.append(info)
                for value in values:value['accepted']=decision['accepted']
                all_rows.extend(values)
                write(ROOT/'reports/ALL_TIMINGS.csv',all_rows);save(ROOT/'reports/TIMING_ATTEMPTS.json',attempts)
                if decision['accepted']:accepted.append(values)
                print('ROUND_DECISION',json.dumps({k:v for k,v in info.items() if k!='samples'}),flush=True)
            observations.append(gpu('timing_end'));save(ROOT/'reports/GPU_OBSERVATIONS.json',observations)
        finally:monitor.close()
        times=[v for round_values in accepted for v in round_values]
        if times:write(ROOT/'reports/TIMINGS.csv',times)
        save(ROOT/'reports/TIMING_ACCEPTANCE.json',dict(complete=len(accepted)==3,accepted_rounds=len(accepted),accepted_rows=len(times),
            expected_rows=792*7*3,attempts=len(attempts),criterion='independent pmon all-external-process activity; whole rounds; no latency-based filtering',resolution_seconds=1))
    assert all(sha(n)==h for n,h in protected.items());save(ROOT/'reports/INPUTS_AFTER.json',{n:sha(n) for n in protected})
    before=json.loads((ROOT/'reports/PRODUCTION_BEFORE.json').read_text());after={n:dict(bytes=Path(n).stat().st_size,sha256=sha(n)) for n in before}
    assert before==after;save(ROOT/'reports/PRODUCTION_AFTER.json',after)
    save(ROOT/'reports/FINAL_INTEGRITY.json',dict(passed=True,production_files=len(before),source_unchanged=all(sha(ROOT/'source'/n)==h for n,h in source.items()),
        input_unchanged=True,images_saved=0,training_runs=0,production_updated=False,commit_push=False))
    state.update(state='complete' if len(accepted)==3 else 'quality_complete_timing_blocked',accepted_rounds=len(accepted),completed_unix=time.time())
    save(ROOT/'STATUS.json',state);print(json.dumps(state),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        p=ROOT/'STATUS.json';s=json.loads(p.read_text()) if p.exists() else {};s.update(state='failed',error=repr(e));save(p,s);raise
