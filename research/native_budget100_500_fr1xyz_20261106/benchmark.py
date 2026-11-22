"""native v4固定小预算：真Alpha点数、原始PSNR和独立的处理／渲染延迟。"""
from common import *
import fcntl
import importlib.util
import math
import random
import shutil
from collections import defaultdict
import numpy as np
import torch
from budget_viewer import BudgetScene
from deployment_runtime import DeploymentRenderer
from renderer import render_prepared
from utils import MiniCam,focal2fov

BUDGETS=[100,200,300,400,500]


def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def bitwise(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))


def counted(binding,camera,data,bg):
    inputs=data['inputs'];empty=torch.Tensor([]);value=lambda k:inputs[k] if inputs[k] is not None else empty
    return binding.render_with_counts(bg,value('means3D'),value('colors_precomp'),value('opacities'),value('scales'),value('rotations'),1.,
        value('cov3D_precomp'),camera.world_view_transform,camera.full_proj_transform,math.tan(camera.FoVx*.5),math.tan(camera.FoVy*.5),
        camera.image_height,camera.image_width,value('dc'),value('shs'),data['sh_degree'],camera.camera_center,False,False,True,binding.Mode.Inference)


def historical(scene):
    folder=ROOT/'reference';counts=read(folder/'CARTGS_COUNTS.csv')
    quality=[v for v in read(folder/'CARTGS_QUALITY.csv') if v['method']=='cartgs_full']
    timing=[v for v in read(folder/'CARTGS_TIMINGS.csv') if v['method']=='cartgs_full']
    assert len(counts)==len(quality)==len(scene.views)==792 and len(timing)==2376
    c={int(v['frame_index']):v for v in counts};q={int(v['frame']):v for v in quality};t=defaultdict(list)
    for v in timing:t[int(v['frame'])].append(v)
    rows=[]
    for f,v in enumerate(scene.views):
        a,b=c[f],q[f];assert abs(float(a['rgb_timestamp'])-v['timestamp'])<1e-6 and abs(float(b['timestamp'])-v['timestamp'])<1e-6
        assert str(Path(v['image']).resolve())==a['rgb_path'] and int(a['total_gaussians'])==int(b['K'])==35301
        assert int(a['projected_gaussians'])==int(b['projected_K']) and a['color_radii_bitwise_equal']=='True'
        assert len(t[f])==3 and {int(x['repeat']) for x in t[f]}=={0,1,2}
        assert all(abs(float(x['timestamp'])-v['timestamp'])<1e-6 for x in t[f])
        rows.append(dict(frame=f,rgb_index=int(a['rgb_index']),timestamp=v['timestamp'],K=35301,projected_K=int(a['projected_gaussians']),
            contributing_K=int(a['contributing_gaussians']),psnr=float(b['psnr']),prepare_ms=0.,
            render_ms=float(np.mean([float(x['render_ms']) for x in t[f]])),total_ms=float(np.mean([float(x['total_ms']) for x in t[f]]))))
    write(ROOT/'evaluation/CARTGS_PER_VIEW.csv',rows);write(ROOT/'reference/CARTGS_TIMINGS_ONLY.csv',timing)
    return rows


@torch.no_grad()
def main():
    lock=(ROOT/'reports/run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (ROOT/'STATUS.json').exists()
    state=dict(state='loading',process=identity(),started_unix=time.time());save(ROOT/'STATUS.json',state);started=time.monotonic()
    torch.set_num_threads(2);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    protected={str(p):sha(p) for p in CHECKPOINT.rglob('*') if p.is_file()};trajectory=CHECKPOINT.parent.parent/'CameraTrajectory_TUM.txt';protected[str(trajectory)]=sha(trajectory)
    runtime_source=json.loads((ROOT/'reports/RUNTIME_SOURCE_HASHES.json').read_text());assert all(sha(ROOT/'source'/n)==h for n,h in runtime_source.items())
    historical_inputs=json.loads((ROOT/'reports/HISTORICAL_INPUTS.json').read_text());assert all(sha(n)==h for n,h in historical_inputs.items())
    native_id=json.loads((ROOT/'reference/NATIVE_BINARY_IDENTITY.json').read_text());assert all(sha(ROOT/'source/python'/n)==h for n,h in native_id['sources'].items())
    scene=BudgetScene(CHECKPOINT,DATASET,100,0,None,'pytorch');scene.model.eval()
    fine=json.loads((CHECKPOINT/'real_lod_state.json').read_text());assert fine['view_budget_conditioning']==2 and abs(fine['view_budget_li_weight']-.1)<1e-6
    assert scene.renderer.total==13298 and len(scene.model.anchors.centers)==2790
    historical(scene);observations=[gpu('before_load')];state.update(state='compiling');save(ROOT/'STATUS.json',state)
    load_start=time.perf_counter();runtime=DeploymentRenderer.from_checkpoint(CHECKPOINT,0);torch.cuda.synchronize();load_seconds=time.perf_counter()-load_start
    assert runtime._ext.__name__=='mycartgs_deployment_fp32_v4' and runtime.backend=='deployment_cpp'
    binary=Path(runtime._ext.__file__);assert ROOT/'build' in binary.parents
    c=scene.calibration;cameras=[MiniCam(c['width'],c['height'],focal2fov(c['fx'],c['width']),focal2fov(c['fy'],c['height']),np.linalg.inv(np.asarray(v['pose']))) for v in scene.views]
    p=Path('/home/prmi/zwj/experiments/fr1xyz_viewer_budget_comparison_20261105/build/fr1xyz_production_counts.so')
    spec=importlib.util.spec_from_file_location('fr1xyz_production_counts',p);binding=importlib.util.module_from_spec(spec);spec.loader.exec_module(binding);protected[str(p)]=sha(p)
    import diff_gaussian_rasterization
    raster=Path(diff_gaussian_rasterization._C.__file__);protected[str(raster)]=sha(raster)
    save(ROOT/'reports/INPUTS_BEFORE.json',protected)
    save(ROOT/'reports/ENVIRONMENT.json',dict(checkpoint=str(CHECKPOINT),N=13298,anchors=2790,frames=792,seed=0,li_weight=.1,joint_schema=2,budgets=BUDGETS,
        backend=runtime.backend,binary=dict(path=str(binary),sha256=sha(binary)),native_source_identity=native_id['sources'],count_binding=str(p),
        width=c['width'],height=c['height'],torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),tf32=False,
        load_compile_seconds=load_seconds,cartgs_history=json.loads((ROOT/'reference/CARTGS_ENVIRONMENT.json').read_text())['vanilla'],
        timing_scope='ours is fresh native v4; cartgs_full reuses historical 3x792 samples; not same GPU load',images_saved=0,training_runs=0))
    def prepare(frame,budget):
        data,ids,stats=runtime.prepare(cameras[frame],budget=budget)
        assert len(ids)==min(budget,stats['eligible_candidates']) and stats['requested_budget']==budget and not stats['cache_hit']
        return data,ids,stats
    def call(frame,budget):
        torch.cuda.synchronize();a=time.perf_counter();data,ids,stats=prepare(frame,budget);b=time.perf_counter()
        render_prepared(cameras[frame],data,scene.background);torch.cuda.synchronize();c=time.perf_counter()
        return dict(frame=frame,budget=budget,K=len(ids),candidate_N=stats['candidate_gaussians'],eligible_N=stats['eligible_candidates'],network_ratio=stats['ratio'],
            shortfall=stats['budget_shortfall'],prepare_ms=(b-a)*1000,render_ms=(c-b)*1000,total_ms=(c-a)*1000)
    with torch.inference_mode():
        state.update(state='smoke');save(ROOT/'STATUS.json',state);parity=[]
        for f in [0,396,791]:
            for budget in BUDGETS:
                data,ids,stats=prepare(f,budget);normal=render_prepared(cameras[f],data,scene.background,return_radii=True);output=counted(binding,cameras[f],data,scene.background)
                assert bitwise(normal['render'],output[2]) and torch.equal(normal['radii'],output[3])
                parity.append(dict(frame=f,budget=budget,render_and_radii_bitwise=True))
        save(ROOT/'reports/COUNT_RENDERER_PARITY.json',dict(passed=True,cases=parity))
        for f in np.linspace(0,791,64,dtype=int):
            for budget in BUDGETS:call(int(f),budget)
        torch.cuda.reset_peak_memory_stats();times=[]
        with (ROOT/'reports/TIMINGS.csv').open('w',newline='') as stream:
            writer=None
            for repeat in range(3):
                frames=list(range(792))
                if repeat==1:frames.reverse()
                if repeat==2:random.Random(0).shuffle(frames)
                observations.append(gpu(f'round_{repeat}_start'))
                for index,f in enumerate(frames):
                    shift=(index+repeat)%5
                    for budget in BUDGETS[shift:]+BUDGETS[:shift]:
                        v=call(f,budget);v.update(repeat=repeat,order=index);times.append(v)
                        if writer is None:writer=csv.DictWriter(stream,fieldnames=list(v));writer.writeheader()
                        writer.writerow(v)
                    if index%100==0 or index==791:
                        stream.flush();state.update(state='timing',repeat=repeat,frame=index+1,rows=len(times),seconds=time.monotonic()-started);save(ROOT/'STATUS.json',state);print(json.dumps(state),flush=True)
                observations.append(gpu(f'round_{repeat}_end'))
        assert len(times)==11880
        save(ROOT/'reports/PEAK_MEMORY.json',dict(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved()))
        quality=[]
        for frame,v in enumerate(scene.views):
            gt=scene.ground_truth(frame)
            for budget in BUDGETS:
                data,ids,stats=prepare(frame,budget);output=counted(binding,cameras[frame],data,scene.background)
                image=output[2].clamp(0,1);mse=(image-gt).square().mean();psnr=float(-10*torch.log10(mse));assert math.isfinite(psnr) and torch.isfinite(output[2]).all()
                projected,contributing=int((output[3]>0).sum()),int((output[5]>0).sum());assert 0<=contributing<=projected<=len(ids)
                quality.append(dict(frame=frame,timestamp=v['timestamp'],image=v['image'],budget=budget,N=13298,K=len(ids),candidate_N=stats['candidate_gaussians'],eligible_N=stats['eligible_candidates'],
                    network_ratio=stats['ratio'],below_training_min=stats['ratio']<runtime.knots[0]-1e-8,shortfall=stats['budget_shortfall'],
                    projected_K=projected,contributing_K=contributing,psnr=psnr,photo_l1=float((image-gt).abs().mean())))
            if frame%50==0 or frame==791:
                state.update(state='quality',frame=frame+1,quality_rows=len(quality),seconds=time.monotonic()-started);save(ROOT/'STATUS.json',state);print(json.dumps(state),flush=True)
        assert len(quality)==3960;write(ROOT/'evaluation/OURS_PER_VIEW.csv',quality)
    assert all(sha(n)==h for n,h in protected.items()) and all(sha(n)==h for n,h in historical_inputs.items())
    assert all(sha(ROOT/'source'/n)==h for n,h in runtime_source.items())
    save(ROOT/'reports/INPUTS_AFTER.json',{n:sha(n) for n in protected})
    old=json.loads((ROOT/'reports/PRODUCTION_BEFORE.json').read_text());after={n:dict(bytes=Path(n).stat().st_size,sha256=sha(n)) for n in old};assert old==after;save(ROOT/'reports/PRODUCTION_AFTER.json',after)
    observations.append(gpu('complete'));save(ROOT/'reports/GPU_OBSERVATIONS.json',observations)
    save(ROOT/'reports/FINAL_INTEGRITY.json',dict(passed=True,protected_production_files=len(old),changed_files=[],source_unchanged=True,model_unchanged=True,historical_inputs_unchanged=True,
        fresh_timing_rows=11880,fresh_quality_rows=3960,reused_cartgs_views=792,reused_cartgs_timing_rows=2376,renderer_bitwise_cases=15,training_runs=0,render_images_saved=0,
        production_updated=False,commit_push=False,free_GiB=shutil.disk_usage(ROOT).free/2**30))
    state.update(state='complete',seconds=time.monotonic()-started);save(ROOT/'STATUS.json',state);print('NATIVE_BUDGET100_500_COMPLETE',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        p=ROOT/'STATUS.json';v=json.loads(p.read_text()) if p.exists() else {};v.update(state='failed',error=repr(error));save(p,v);raise
