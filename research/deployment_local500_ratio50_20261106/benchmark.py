"""相同默认模型、相机、预算的三后端配对速度与质量复测。"""
from common import *
import fcntl
import math
import random
import shutil
import numpy as np
import torch
from budget_viewer import BudgetScene
from anchor_budget import FixedBudgetRenderer
from deployment_runtime import DeploymentRenderer
from renderer import render_prepared
from utils import MiniCam,focal2fov

REQUESTS={'points500':dict(budget=500),'ratio50':dict(ratio_budget=.5)}
METHODS=[(tag,case) for tag in ['pytorch','cuda','deployment'] for case in REQUESTS]+[('full','full')]


def bitwise(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))


@torch.no_grad()
def main():
    lock=(ROOT/'reports/run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'STATUS.json').exists()
    state=dict(state='loading',process=identity(),started_unix=time.time());save(ROOT/'STATUS.json',state)
    started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    protected={str(p):sha(p) for p in CHECKPOINT.rglob('*') if p.is_file()}
    protected[str(CHECKPOINT.parent.parent/'CameraTrajectory_TUM.txt')]=sha(CHECKPOINT.parent.parent/'CameraTrajectory_TUM.txt')
    source=json.loads((ROOT/'reports/SOURCE_HASHES.json').read_text());assert all(sha(ROOT/'source'/n)==h for n,h in source.items())
    fine=json.loads((CHECKPOINT/'real_lod_state.json').read_text())
    assert fine['lod_steps']==11580 and fine['view_budget_conditioning']==2 and abs(fine['view_budget_li_weight']-.1)<1e-6
    observations=[gpu('before_load')]
    scene=BudgetScene(CHECKPOINT,DATASET,500,0,None,'pytorch');scene.model.eval()
    assert len(scene.views)==792 and scene.renderer.total==13298 and len(scene.model.anchors.centers)==2790
    assert int(scene.model.joint_view_conditioning_schema)==2 and not hasattr(scene.model.spatial_compensation,'geometry_head_schema')
    state.update(state='compiling',seconds=time.monotonic()-started);save(ROOT/'STATUS.json',state)
    engines={'pytorch':scene.renderer,'cuda':FixedBudgetRenderer(scene.model,scene.gaussians,scene.config,0,'cuda')}
    torch.cuda.synchronize();memory_before=torch.cuda.memory_allocated();loaded=time.perf_counter()
    engines['deployment']=DeploymentRenderer.from_checkpoint(CHECKPOINT,0)
    torch.cuda.synchronize();load_seconds=time.perf_counter()-loaded;memory_after=torch.cuda.memory_allocated()
    c=scene.calibration;cameras=[MiniCam(c['width'],c['height'],focal2fov(c['fx'],c['width']),
        focal2fov(c['fy'],c['height']),np.linalg.inv(np.asarray(v['pose']))) for v in scene.views]
    binary={}
    for tag,mod in [('v3',engines['cuda'].inference.ext),('deployment',engines['deployment']._ext)]:
        p=Path(mod.__file__);binary[tag]=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p));assert ROOT/'build' in p.parents
    import diff_gaussian_rasterization
    raster=Path(diff_gaussian_rasterization._C.__file__);protected[str(raster)]=sha(raster)
    save(ROOT/'reports/INPUTS_BEFORE.json',protected)
    save(ROOT/'reports/ENVIRONMENT.json',dict(checkpoint=str(CHECKPOINT),source_commit='26c355b32bf373a75877fc4605a4e41b8de9b47f',
        N=13298,anchors=2790,frames=792,width=c['width'],height=c['height'],joint_schema=2,li_weight=.1,seed=0,tf32=False,threads=2,
        torch=torch.__version__,cuda=torch.version.cuda,python=sys.version,device=torch.cuda.get_device_name(),
        binaries=binary,rasterizer=str(raster),deployment_load_compile_seconds=load_seconds,load_all_seconds=time.monotonic()-started,
        deployment_persistent_increment_bytes=memory_after-memory_before,build_directory=str(ROOT/'build'),source_directory=str(ROOT/'source/python'),
        quality_gates=dict(mean_delta_min=-.01,p01_delta_min=-.05,worst_delta_min=-.1),training_runs=0,images_saved=0))

    def prepare(tag,case,frame):
        if tag=='full':return scene.full_data,scene.full_ids,dict(candidate_gaussians=13298,eligible_candidates=13298,requested_budget=13298,budget_shortfall=0,ratio=None)
        kw=dict(REQUESTS[case],view_conditioned=True)
        if tag!='deployment':kw['measure_time']=False
        data,ids,stats=engines[tag].prepare(cameras[frame],**kw)
        expected=500 if case=='points500' else math.floor(.5*stats['candidate_gaussians'])
        assert len(ids)==min(expected,stats['eligible_candidates']) and stats['requested_budget']==expected and not stats['cache_hit']
        return data,ids,stats

    events=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
    def call(tag,case,frame,event=False):
        torch.cuda.synchronize();t=time.perf_counter()
        if event:events[0].record()
        data,ids,stats=prepare(tag,case,frame);mid=time.perf_counter()
        if event:events[1].record()
        image=render_prepared(cameras[frame],data,scene.background,return_radii=True)
        if event:events[2].record()
        torch.cuda.synchronize();end=time.perf_counter()
        row=dict(backend=tag,case=case,frame=frame,K=len(ids),candidate_N=stats['candidate_gaussians'],eligible_N=stats['eligible_candidates'],
            requested_K=stats['requested_budget'],network_ratio=stats['ratio'],budget_shortfall=stats['budget_shortfall'],
            inference_ms=0. if tag=='full' else (mid-t)*1000,render_ms=(end-t if tag=='full' else end-mid)*1000,total_ms=(end-t)*1000)
        if event:row.update(inference_stream_ms=events[0].elapsed_time(events[1]),render_stream_ms=events[1].elapsed_time(events[2]),total_stream_ms=events[0].elapsed_time(events[2]))
        return row,image,ids

    with torch.inference_mode():
        state.update(state='smoke');save(ROOT/'STATUS.json',state);smoke=[]
        for f in [0,198,396,594,791]:
            for case in REQUESTS:
                a,ia,sa=prepare('pytorch',case,f);b,ib,sb=prepare('cuda',case,f);d,ids,sd=prepare('deployment',case,f)
                assert torch.equal(ia,ib) and len(ia)==len(ids) and sa['candidate_gaussians']==sd['candidate_gaussians']
                for key,value in a['inputs'].items():
                    if isinstance(value,torch.Tensor):assert bitwise(value,b['inputs'][key]),key
                x=render_prepared(cameras[f],b,scene.background)['render'];y=render_prepared(cameras[f],d,scene.background)['render']
                assert torch.isfinite(y).all()
                smoke.append(dict(frame=f,case=case,pytorch_cuda_packet_bitwise=True,ids_same=torch.equal(ib,ids),K=len(ids),image_max_abs=float((x-y).abs().max())))
        save(ROOT/'reports/SMOKE.json',dict(passed=True,cases=smoke))
        for f in np.linspace(0,791,64,dtype=int):
            for tag,case in METHODS:call(tag,case,int(f))
        torch.cuda.reset_peak_memory_stats();rows=[]
        with (ROOT/'reports/TIMINGS.csv').open('w',newline='') as stream:
            writer=None
            for repeat in range(3):
                order=list(range(792))
                if repeat==1:order.reverse()
                if repeat==2:random.Random(0).shuffle(order)
                observations.append(gpu(f'round_{repeat}_start'))
                for index,f in enumerate(order):
                    shift=(index+repeat)%len(METHODS)
                    for tag,case in METHODS[shift:]+METHODS[:shift]:
                        r,image,ids=call(tag,case,f);r.update(repeat=repeat,order=index);rows.append(r)
                        if writer is None:writer=csv.DictWriter(stream,fieldnames=list(r));writer.writeheader()
                        writer.writerow(r)
                    if index%100==0 or index==791:
                        stream.flush();state.update(state='wall_timing',repeat=repeat,frame=index+1,rows=len(rows),seconds=time.monotonic()-started)
                        save(ROOT/'STATUS.json',state);print(json.dumps(state),flush=True)
                observations.append(gpu(f'round_{repeat}_end'))
        assert len(rows)==16632
        save(ROOT/'reports/PEAK_MEMORY.json',dict(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved(),deployment_persistent_increment_bytes=memory_after-memory_before))
        event_rows=[]
        for f in np.linspace(0,791,96,dtype=int):
            for tag,case in METHODS:
                r,image,ids=call(tag,case,int(f),True);event_rows.append(r)
        write(ROOT/'reports/CUDA_EVENTS.csv',event_rows)
        component_benchmark(engines,cameras[396])
        quality(scene,prepare,cameras,state,started)
        state.update(state='profiling');save(ROOT/'STATUS.json',state)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            for f in [0,264,528,791]:
                for tag,case in METHODS:
                    with torch.profiler.record_function('METHOD/'+tag+'/'+case):call(tag,case,f)
        prof.export_chrome_trace(str(ROOT/'reports/PROFILE.json'))
    for engine in [engines['pytorch'],engines['cuda']]:engine.check_state()
    after={p:sha(p) for p in protected};assert after==protected
    save(ROOT/'reports/INPUTS_AFTER.json',after)
    assert all(sha(ROOT/'source'/n)==h for n,h in source.items())
    before=json.loads((ROOT/'reports/PRODUCTION_BEFORE.json').read_text())
    after={p:dict(bytes=Path(p).stat().st_size,sha256=sha(p)) for p in before};assert before==after
    save(ROOT/'reports/PRODUCTION_AFTER.json',after)
    observations.append(gpu('complete'));save(ROOT/'reports/GPU_OBSERVATIONS.json',observations)
    save(ROOT/'reports/FINAL_INTEGRITY.json',dict(passed=True,protected_production_files=len(before),changed_files=[],checkpoint_unchanged=True,source_unchanged=True,
        timed_calls=16632,event_calls=672,quality_calls=1584,training_runs=0,render_images_saved=0,production_updated=False,commit_push=False,free_GiB=shutil.disk_usage(ROOT).free/2**30))
    state.update(state='complete',seconds=time.monotonic()-started);save(ROOT/'STATUS.json',state);print('BENCHMARK_COMPLETE',flush=True)


def component_benchmark(engines,cam):
    ref=engines['cuda'];fast=engines['deployment'];b=ref.inference;f=fast._base
    code=b.view_code(cam);fast_code=fast.view_code(cam)
    rows,_,available=b.candidates(b.frustum(cam.full_proj_transform,ref.occupied));records=[]
    events=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
    for case in REQUESTS:
        k=min(500 if case=='points500' else math.floor(.5*len(rows)),available);ratio=500/len(rows) if case=='points500' else .5
        logits=b.logits(rows,ratio,code,True);ids=rows[b.select(b.scores(logits,ref.noise,rows),b.eligible[rows],k)]
        controls=b.controls(ratio);weights,ratio_code=fast._controls(ratio)
        functions={'selector':dict(cuda=lambda:b.scores(b.logits(rows,ratio,code,True),ref.noise,rows),
            deployment=lambda:fast._ext.selector(f.selector_static,rows,f.owners,fast_code,fast._reference.noise,fast._selector,ratio,True,f.joint_schema)),
            'compensation_decode':dict(cuda=lambda:b._compute(ids,controls,True,code,True),
            deployment=lambda:fast._ext.compensate(f.offset,ids,f.owners,f.reference,f.tables,fast._compensator,f.compensated_master,
                fast_code,fast._view_weight,weights,ratio_code,True,f.joint_schema,True))}
        for stage,methods in functions.items():
            for fn in methods.values():
                for _ in range(32):fn()
            for repeat in range(200):
                for tag in (list(methods) if repeat%2==0 else list(reversed(methods))):
                    torch.cuda.synchronize();start=time.perf_counter();events[0].record();methods[tag]();events[1].record();events[1].synchronize()
                    records.append(dict(case=case,stage=stage,backend=tag,repeat=repeat,N=len(rows),K=k,network_ratio=ratio,
                        wall_ms=(time.perf_counter()-start)*1000,stream_ms=events[0].elapsed_time(events[1])))
    write(ROOT/'reports/COMPONENTS.csv',records)


def quality(scene,prepare,cameras,state,started):
    records=[];perceptual=[]
    from torchmetrics.functional.image import structural_similarity_index_measure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    lpips=LearnedPerceptualImagePatchSimilarity(normalize=True,reduction='none').cuda()
    for frame in range(792):
        gt=scene.ground_truth(frame)
        for case in REQUESTS:
            a,ia,sa=prepare('cuda',case,frame);b,ib,sb=prepare('deployment',case,frame)
            assert len(ia)==len(ib) and sa['candidate_gaussians']==sb['candidate_gaussians']
            x=render_prepared(cameras[frame],a,scene.background,return_radii=True);y=render_prepared(cameras[frame],b,scene.background,return_radii=True)
            xi,yi=x['render'],y['render'];diff=(xi-yi).abs();same=torch.equal(ia,ib)
            psnr=lambda image:float(-10*torch.log10((image.clamp(0,1)-gt).square().mean()))
            pa,pb=psnr(xi),psnr(yi);assert math.isfinite(pa) and math.isfinite(pb) and torch.isfinite(yi).all()
            if same:retention=1.
            else:retention=float(torch.isin(ib,ia).float().mean()) if len(ib) else 1.
            row=dict(case=case,frame=frame,timestamp=scene.views[frame]['timestamp'],candidate_N=sa['candidate_gaussians'],K=len(ib),
                ratio=sa['ratio'],base_psnr=pa,deployment_psnr=pb,delta_psnr=pb-pa,ids_same=same,id_retention=retention,
                image_max_abs=float(diff.max()),image_mean_abs=float(diff.mean()),image_mse=float(diff.square().mean()),
                radii_same=torch.equal(x['radii'],y['radii']),projected_K=int((y['radii']>0).sum()))
            for name in ['means3D','scales','rotations','opacities','dc']:
                row[name+'_max_abs']=float((a['inputs'][name]-b['inputs'][name]).abs().max()) if same and len(ia) else None
            records.append(row)
            if frame%12==0:
                # Same-source PyTorch direct check, including SSIM/LPIPS on 66 matched views.
                py,ip,sp=prepare('pytorch',case,frame);assert torch.equal(ip,ia)
                for key,value in py['inputs'].items():
                    if isinstance(value,torch.Tensor):assert bitwise(value,a['inputs'][key]),key
                image=render_prepared(cameras[frame],py,scene.background)['render'];assert bitwise(image,xi)
                batch=torch.stack([xi.clamp(0,1),yi.clamp(0,1)]);target=gt.unsqueeze(0).expand_as(batch)
                ss=structural_similarity_index_measure(batch,target,data_range=1.,reduction='none').cpu().tolist()
                ll=lpips(batch,target).reshape(-1).cpu().tolist();lpips.reset()
                perceptual.append(dict(case=case,frame=frame,base_ssim=ss[0],deployment_ssim=ss[1],delta_ssim=ss[1]-ss[0],
                    base_lpips=ll[0],deployment_lpips=ll[1],delta_lpips=ll[1]-ll[0],pytorch_cuda_image_bitwise=True))
        if frame%50==0 or frame==791:
            state.update(state='quality',frame=frame+1,quality_rows=len(records),seconds=time.monotonic()-started);save(ROOT/'STATUS.json',state);print(json.dumps(state),flush=True)
    write(ROOT/'evaluation/PER_FRAME.csv',records);write(ROOT/'evaluation/PERCEPTUAL.csv',perceptual)
    assert len(records)==1584 and len(perceptual)==132
    gates=[]
    for case in REQUESTS:
        values=[v for v in records if v['case']==case];delta=np.asarray([v['delta_psnr'] for v in values])
        item=dict(case=case,mean_delta=float(delta.mean()),p01_delta=float(np.percentile(delta,1)),worst_delta=float(delta.min()),
            ids_same_fraction=float(np.mean([v['ids_same'] for v in values])),minimum_id_retention=min(v['id_retention'] for v in values))
        item['passed']=item['mean_delta']>=-.01 and item['p01_delta']>=-.05 and item['worst_delta']>=-.1;gates.append(item)
    save(ROOT/'reports/QUALITY_GATES.json',dict(passed=all(v['passed'] for v in gates),cases=gates))
    assert all(v['passed'] for v in gates),'部署质量门槛失败，保留现场'


if __name__=='__main__':
    try:main()
    except Exception as error:
        path=ROOT/'STATUS.json';state=json.loads(path.read_text()) if path.exists() else {}
        state.update(state='failed',error=repr(error));save(path,state);raise
