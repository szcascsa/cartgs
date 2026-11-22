"""同一默认模型的局部50%与500点配对测时，使用提交版源码。"""
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OLD=Path('/home/prmi/zwj/experiments/global_view_joint_lod_20261105')
ORIGINAL=Path('/home/prmi/zwj/experiments/original_cartgs_eval_fr1_full8900_20261103_01/source/python')
DEFAULT=Path('/home/prmi/zwj/experiments/view_conditioned_li_20261106/runs/full_both_li/23560_shutdown/ply')
DATASET=Path('/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz')
sys.path[:0]=[str(ROOT/'source/python'),str(OLD/'evaluation/python'),str(OLD/'evaluation/python/diff-gaussian-rasterization')]
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import numpy as np
import torch
from budget_viewer import BudgetScene,load_views
from gaussian_model import GaussianModel
from renderer import render_prepared,prepare_render_data
from utils import MiniCam,focal2fov
import diff_gaussian_rasterization

METHODS=['cartgs_full','cartgs_full_cached','default_ratio50','default_500','default_full']


def save(name,value):
    path=ROOT/name;temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False));temp.replace(path)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4194304),b''):digest.update(chunk)
    return digest.hexdigest()


def module(name,path,package=False):
    spec=importlib.util.spec_from_file_location(name,path,submodule_search_locations=[str(Path(path).parent)] if package else None)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value


def gpu_state(label):
    result=dict(label=label,unix=time.time())
    for key,cmd in [('gpu',['nvidia-smi','--query-gpu=name,uuid,memory.used,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem,power.draw','--format=csv,noheader']),
                    ('processes',['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'])]:
        result[key]=subprocess.check_output(cmd,text=True).strip()
    return result


def cameras(calibration,views):
    c=calibration
    return [MiniCam(c['width'],c['height'],focal2fov(c['fx'],c['width']),focal2fov(c['fy'],c['height']),np.linalg.inv(np.asarray(v['pose']))) for v in views]


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    started=time.monotonic();assert not (ROOT/'reports/TIMINGS.csv').exists()
    state=dict(state='loading',pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
               boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip());save('STATUS.json',state)
    protected={}
    old_manifest=json.loads(Path('/home/prmi/zwj/experiments/view_conditioned_li_20261106/reports/PRODUCTION_BEFORE.json').read_text())
    for name in old_manifest:protected[name]=sha(name)
    summary=json.loads(Path('/home/prmi/zwj/experiments/vanilla_fr1xyz_render_counts_20261105/reports/SUMMARY.json').read_text())
    vanilla=next(v for v in summary if v['mode']=='rgbd' and v['best_full_psnr_repeat'])
    vcheckpoint=Path(vanilla['checkpoint'])/'ply'
    for checkpoint in [DEFAULT,vcheckpoint]:
        for p in checkpoint.rglob('*'):
            if p.is_file():protected[str(p)]=sha(p)
        p=checkpoint.parent.parent/'CameraTrajectory_TUM.txt';protected[str(p)]=sha(p)
    for p in [*list((ROOT/'source').rglob('*.py')),ORIGINAL/'renderer.py',ORIGINAL/'utils.py',
              ORIGINAL/'diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py',
              *list((ORIGINAL/'diff-gaussian-rasterization/diff_gaussian_rasterization').glob('*.so')),
              Path(diff_gaussian_rasterization._C.__file__)]:protected[str(p)]=sha(p)
    save('reports/INPUTS_BEFORE.json',protected)
    loading=time.perf_counter();scenes={}
    for name,checkpoint in [('default',DEFAULT)]:
        scenes[name]=BudgetScene(checkpoint,DATASET,500,0,None,'pytorch');scenes[name].model.eval()
    assert hasattr(scenes['default'].model,'joint_view_conditioning_schema')
    assert int(scenes['default'].model.joint_view_conditioning_schema)==2
    assert scenes['default'].renderer.total==13298
    vcal,vviews=load_views(vcheckpoint,DATASET)
    own=scenes['default'];assert vcal==own.calibration
    keys=lambda v:[(round(x['timestamp'],6),str(Path(x['image']).resolve())) for x in v]
    assert len(own.views)==len(vviews)==792 and keys(own.views)==keys(vviews)
    cams={name:cameras(scene.calibration,scene.views) for name,scene in scenes.items()}
    cams['vanilla']=cameras(vcal,vviews)
    pc=GaussianModel(3);pc.load_ply(vanilla['ply_path']);assert len(pc.get_xyz)==35301
    bg=torch.zeros(3,device='cuda');vdata=prepare_render_data(pc)
    original_raster=module('cartgs_original_raster',ORIGINAL/'diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py',True)
    original_renderer=module('cartgs_original_renderer',ORIGINAL/'renderer.py')
    # The original module uses absolute imports; bind its two classes to its original extension.
    original_renderer.GaussianRasterizationSettings=original_raster.GaussianRasterizationSettings
    original_renderer.GaussianRasterizer=original_raster.GaussianRasterizer
    torch.cuda.synchronize();loading_seconds=time.perf_counter()-loading
    state.update(state='validating',frame=0);save('STATUS.json',state)
    environment=dict(torch=torch.__version__,cuda=torch.version.cuda,python=sys.version,device=torch.cuda.get_device_name(),
        original_extension=str(original_raster._C.__file__),current_extension=str(diff_gaussian_rasterization._C.__file__),
        python_source=str(ROOT/'source/python'),default_checkpoint=str(DEFAULT),local_ratio=.5,fixed_budget=500,vanilla=vanilla,
        model_load_and_camera_build_seconds=loading_seconds,tf32=False,seed=0,threads=2,
        width=vcal['width'],height=vcal['height'],frames=792,camera_protocol='同RGB时间戳，各模型原生SLAM位姿；500点与50%使用同一模型同一相机')
    save('reports/ENVIRONMENT.json',environment)

    def original_cached(camera):
        settings=original_raster.GaussianRasterizationSettings(image_height=camera.image_height,image_width=camera.image_width,
            tanfovx=math.tan(camera.FoVx*.5),tanfovy=math.tan(camera.FoVy*.5),bg=bg,scale_modifier=1.,
            viewmatrix=camera.world_view_transform,projmatrix=camera.full_proj_transform,sh_degree=pc.active_sh_degree,
            campos=camera.camera_center,prefiltered=False,debug=False)
        image,radii=original_raster.GaussianRasterizer(raster_settings=settings)(**vdata['inputs'])
        return dict(render=image,radii=radii)

    def call(method,frame,events=False):
        torch.cuda.synchronize();before=time.perf_counter()
        if events:ev[0].record()
        stats={};ids=None
        if method in ['default_ratio50','default_500']:
            which='default';scene=own
            request={'ratio_budget':.5} if method=='default_ratio50' else {'budget':500}
            data,ids,stats=scene.renderer.prepare(cams[which][frame],view_conditioned=True,**request)
            infer_done=time.perf_counter()
            if events:ev[1].record()
            out=render_prepared(cams[which][frame],data,bg,return_radii=True)
        else:
            infer_done=before
            if events:ev[1].record()
            if method=='cartgs_full':out=original_renderer.render(cams['vanilla'][frame],pc,bg)
            elif method=='cartgs_full_cached':out=original_cached(cams['vanilla'][frame])
            else:out=render_prepared(cams['default'][frame],own.full_data,bg,return_radii=True)
        if events:ev[2].record()
        torch.cuda.synchronize();done=time.perf_counter()
        record=dict(method=method,frame=frame,timestamp=own.views[frame]['timestamp'],inference_ms=(infer_done-before)*1000,
            render_ms=(done-infer_done)*1000,total_ms=(done-before)*1000,
            selector_ms=stats.get('selector_ms',0.),compensation_ms=stats.get('compensation_ms',0.),
            prepare_internal_ms=stats.get('prepare_ms',0.),candidate_N=stats.get('candidate_gaussians',0),eligible_N=stats.get('eligible_candidates',0),
            K=len(ids) if ids is not None else (35301 if method.startswith('cartgs') else 13298),
            network_ratio=stats.get('ratio'),requested_K=stats.get('requested_budget',0),budget_shortfall=stats.get('budget_shortfall',0),
            view_conditioned=stats.get('view_conditioned',False),cache_hit=stats.get('cache_hit',False))
        record['other_prepare_ms']=record['inference_ms']-record['selector_ms']-record['compensation_ms']
        if events:record.update(inference_stream_ms=ev[0].elapsed_time(ev[1]),render_stream_ms=ev[1].elapsed_time(ev[2]),total_stream_ms=ev[0].elapsed_time(ev[2]))
        if ids is not None:
            expected=math.floor(.5*stats['candidate_gaussians']) if method=='default_ratio50' else 500
            assert stats['requested_budget']==expected and len(ids)==min(expected,stats['eligible_candidates'])
            assert stats['budget_shortfall']==expected-len(ids) and not stats['cache_hit'] and stats['view_conditioned']
        return record,out,ids

    ev=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
    cold=[]
    for name in METHODS:
        rec,out,ids=call(name,0);cold.append(rec)
    save('reports/COLD_FIRST_CALLS.json',cold)
    parity=[]
    for f in [0,396,791]:
        a=original_renderer.render(cams['vanilla'][f],pc,bg);b=original_cached(cams['vanilla'][f])
        assert torch.equal(a['render'].view(torch.int32),b['render'].view(torch.int32)) and torch.equal(a['radii'],b['radii'])
        parity.append(dict(frame=f,original_full_vs_cached_bitwise=True))
        for name in ['default_ratio50','default_500']:
            a,im,ids=call(name,f);b,other,ids2=call(name,f)
            assert torch.equal(ids,ids2) and torch.equal(im['render'].view(torch.int32),other['render'].view(torch.int32))
            parity.append(dict(frame=f,method=name,repeat_selection_and_image_bitwise=True,K=len(ids)))
    save('reports/PARITY.json',parity)
    observations=[gpu_state('before_warmup')]
    warm=[int(x) for x in np.linspace(0,791,64)]
    for f in warm:
        for name in METHODS:
            rec,out,ids=call(name,f)
    torch.cuda.reset_peak_memory_stats();rows=[]
    with (ROOT/'reports/TIMINGS.csv').open('w',newline='') as file:
        writer=None
        for repeat in range(3):
            order=list(range(792))
            if repeat==1:order.reverse()
            elif repeat==2:random.Random(0).shuffle(order)
            observations.append(gpu_state('round_'+str(repeat)+'_start'))
            for index,f in enumerate(order):
                methods=METHODS[(index+repeat)%len(METHODS):]+METHODS[:(index+repeat)%len(METHODS)]
                for name in methods:
                    rec,out,ids=call(name,f);rec.update(repeat=repeat,order=index)
                    if writer is None:writer=csv.DictWriter(file,fieldnames=list(rec));writer.writeheader()
                    writer.writerow(rec);rows.append(rec)
                if index%100==0 or index==791:
                    file.flush();state.update(state='wall_benchmark',repeat=repeat,frame=index+1,rows=len(rows),seconds=time.monotonic()-started);save('STATUS.json',state);print(json.dumps(state),flush=True)
            observations.append(gpu_state('round_'+str(repeat)+'_end'))
    peak=dict(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved())
    save('reports/PEAK_MEMORY.json',peak)
    save('reports/GPU_OBSERVATIONS.json',observations)
    event_rows=[]
    for f in [int(x) for x in np.linspace(0,791,96)]:
        for name in METHODS:
            rec,out,ids=call(name,f,True);event_rows.append(rec)
    with (ROOT/'reports/CUDA_EVENTS.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(event_rows[0]));writer.writeheader();writer.writerows(event_rows)
    state.update(state='quality',frame=0);save('STATUS.json',state)
    quality=[]
    for f in range(792):
        target=own.ground_truth(f)
        for name in ['cartgs_full','default_ratio50','default_500','default_full']:
            rec,out,ids=call(name,f)
            image=out['render'].clamp(0,1);psnr=float(-10*torch.log10((image-target).square().mean()))
            assert math.isfinite(psnr)
            quality.append(dict(method=name,frame=f,timestamp=own.views[f]['timestamp'],K=rec['K'],
                projected_K=int((out['radii']>0).sum()),psnr=psnr,candidate_N=rec['candidate_N'],eligible_N=rec['eligible_N'],
                requested_K=rec['requested_K'],budget_shortfall=rec['budget_shortfall'],network_ratio=rec['network_ratio']))
        if f%100==0 or f==791:state.update(frame=f+1);save('STATUS.json',state)
    with (ROOT/'reports/QUALITY_COUNTS.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(quality[0]));writer.writeheader();writer.writerows(quality)
    state.update(state='profiling');save('STATUS.json',state)
    activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities,profile_memory=True) as prof:
        for f in [0,264,528,791]:
            for name in METHODS:
                with torch.profiler.record_function('METHOD/'+name):rec,out,ids=call(name,f)
    prof.export_chrome_trace(str(ROOT/'reports/PROFILE.json'))
    aggregates=[]
    for v in prof.key_averages():
        aggregates.append(dict(name=v.key,count=v.count,self_cpu_us=v.self_cpu_time_total,cpu_us=v.cpu_time_total,
            self_cuda_us=v.self_cuda_time_total,cuda_us=v.cuda_time_total,cpu_memory_bytes=v.cpu_memory_usage,cuda_memory_bytes=v.cuda_memory_usage))
    save('reports/PROFILE_AGGREGATES.json',aggregates)
    after={p:sha(p) for p in protected};assert after==protected
    save('reports/INPUTS_AFTER.json',after)
    for scene in scenes.values():scene.renderer.check_state()
    save('reports/FINAL_INTEGRITY.json',dict(passed=True,protected_files=len(protected),production_files=len(old_manifest),changed_files=[],
        methods=METHODS,frames=792,repeats=3,timed_calls=len(rows),event_calls=len(event_rows),saved_images=0,training_steps=0,
        model_parameters_unchanged=True,production_updated=False,git_committed=False,source_sha=sha(HERE/'benchmark.py'),
        gpu_after=gpu_state('after_all'),seconds=time.monotonic()-started))
    state.update(state='complete',frames=792,rows=len(rows),seconds=time.monotonic()-started);save('STATUS.json',state)
    print(json.dumps(state),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        save('STATUS.json',dict(state='failed',pid=os.getpid(),error=repr(error)));raise
