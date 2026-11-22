"""四档局部ratio：冻结部署质量与prepare／渲染真实墙钟。"""
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

RATIOS=[.1,.3,.5,.8]
CASES=[f'local_{r:g}' for r in RATIOS]+['full']


def bitwise(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))


@torch.no_grad()
def main():
    lock=(ROOT/'reports/run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'STATUS.json').exists()
    state=dict(state='loading',process=identity(),started_unix=time.time());save(ROOT/'STATUS.json',state);started=time.monotonic()
    torch.set_num_threads(2);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    before={str(p):sha(p) for p in CHECKPOINT.rglob('*') if p.is_file()}
    trajectory=CHECKPOINT.parent.parent/'CameraTrajectory_TUM.txt';before[str(trajectory)]=sha(trajectory)
    source=json.loads((ROOT/'reports/RUNTIME_SOURCE_HASHES.json').read_text());assert all(sha(ROOT/'source'/n)==h for n,h in source.items())
    fine=json.loads((CHECKPOINT/'real_lod_state.json').read_text());assert fine['view_budget_conditioning']==2 and abs(fine['view_budget_li_weight']-.1)<1e-6
    scene=BudgetScene(CHECKPOINT,DATASET,500,0,None,'pytorch');scene.model.eval()
    assert len(scene.views)==792 and scene.renderer.total==13298 and len(scene.model.anchors.centers)==2790
    observations=[gpu('before_load')];state.update(state='compiling');save(ROOT/'STATUS.json',state)
    ref=FixedBudgetRenderer(scene.model,scene.gaussians,scene.config,0,'cuda')
    torch.cuda.synchronize();mem0=torch.cuda.memory_allocated();loading=time.perf_counter()
    runtime=DeploymentRenderer.from_checkpoint(CHECKPOINT,0);torch.cuda.synchronize()
    mem1=torch.cuda.memory_allocated();load_seconds=time.perf_counter()-loading
    c=scene.calibration;cameras=[MiniCam(c['width'],c['height'],focal2fov(c['fx'],c['width']),focal2fov(c['fy'],c['height']),np.linalg.inv(np.asarray(v['pose']))) for v in scene.views]
    bank={str((DATASET/v['img_name']).resolve()) for v in json.loads((CHECKPOINT/'cameras.json').read_text())}
    assert sum(str(Path(v['image']).resolve()) in bank for v in scene.views)==72
    binary={}
    for tag,mod in [('v3',ref.inference.ext),('deployment',runtime._ext)]:
        p=Path(mod.__file__);assert ROOT/'build' in p.parents;binary[tag]=dict(path=str(p),sha256=sha(p))
    import diff_gaussian_rasterization
    p=Path(diff_gaussian_rasterization._C.__file__);before[str(p)]=sha(p)
    save(ROOT/'reports/INPUTS_BEFORE.json',before)
    save(ROOT/'reports/ENVIRONMENT.json',dict(checkpoint=str(CHECKPOINT),N=13298,anchors=2790,frames=792,bank_frames=72,nonbank_frames=720,
        width=c['width'],height=c['height'],joint_schema=2,li_weight=.1,seed=0,tf32=False,ratios=RATIOS,training_ratios=list(runtime.knots),
        extrapolation_ratios=[.8],torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),binaries=binary,
        deployment_load_compile_seconds=load_seconds,deployment_persistent_bytes=mem1-mem0,all_load_seconds=time.monotonic()-started))

    def prepare(case,frame,backend='deployment'):
        if case=='full':return scene.full_data,scene.full_ids,dict(candidate_gaussians=13298,eligible_candidates=13298,requested_budget=13298,budget_shortfall=0,ratio=None)
        ratio=float(case.split('_')[1]);kwargs=dict(ratio_budget=ratio,view_conditioned=True)
        engine=runtime if backend=='deployment' else ref if backend=='cuda' else scene.renderer
        if backend!='deployment':kwargs['measure_time']=False
        data,ids,stats=engine.prepare(cameras[frame],**kwargs);expected=math.floor(ratio*stats['candidate_gaussians'])
        assert stats['requested_budget']==expected and len(ids)==min(expected,stats['eligible_candidates']) and not stats['cache_hit']
        return data,ids,stats

    events=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
    def call(case,frame,event=False):
        torch.cuda.synchronize();start=time.perf_counter()
        if event:events[0].record()
        data,ids,stats=prepare(case,frame);mid=time.perf_counter()
        if event:events[1].record()
        image=render_prepared(cameras[frame],data,scene.background,return_radii=True)
        if event:events[2].record()
        torch.cuda.synchronize();end=time.perf_counter()
        v=dict(case=case,ratio=stats['ratio'],frame=frame,N=13298,candidate_N=stats['candidate_gaussians'],eligible_N=stats['eligible_candidates'],
            requested_K=stats['requested_budget'],K=len(ids),shortfall=stats['budget_shortfall'],prepare_ms=0. if case=='full' else (mid-start)*1000,
            render_ms=(end-start if case=='full' else end-mid)*1000,total_ms=(end-start)*1000)
        if event:v.update(prepare_stream_ms=events[0].elapsed_time(events[1]),render_stream_ms=events[1].elapsed_time(events[2]),total_stream_ms=events[0].elapsed_time(events[2]))
        return v

    with torch.inference_mode():
        state.update(state='smoke');save(ROOT/'STATUS.json',state);smoke=[]
        for f in [0,198,396,594,791]:
            for case in CASES[:-1]:
                a,ia,sa=prepare(case,f,'pytorch');b,ib,sb=prepare(case,f,'cuda');d,ids,sd=prepare(case,f)
                assert torch.equal(ia,ib) and len(ids)==len(ib) and sa['candidate_gaussians']==sd['candidate_gaussians']
                for key,value in a['inputs'].items():
                    if isinstance(value,torch.Tensor):assert bitwise(value,b['inputs'][key]),key
                smoke.append(dict(frame=f,case=case,exact_packet_bitwise=True,deployment_ids_same=torch.equal(ib,ids),K=len(ids)))
        save(ROOT/'reports/SMOKE.json',dict(passed=True,cases=smoke))
        for f in np.linspace(0,791,64,dtype=int):
            for case in CASES:call(case,int(f))
        torch.cuda.reset_peak_memory_stats();rows=[]
        with (ROOT/'reports/TIMINGS.csv').open('w',newline='') as stream:
            writer=None
            for repeat in range(3):
                order=list(range(792))
                if repeat==1:order.reverse()
                if repeat==2:random.Random(0).shuffle(order)
                observations.append(gpu(f'round_{repeat}_start'))
                for index,f in enumerate(order):
                    shift=(repeat+index)%len(CASES)
                    for case in CASES[shift:]+CASES[:shift]:
                        v=call(case,f);v.update(repeat=repeat,order=index);rows.append(v)
                        if writer is None:writer=csv.DictWriter(stream,fieldnames=list(v));writer.writeheader()
                        writer.writerow(v)
                    if index%100==0 or index==791:
                        stream.flush();state.update(state='wall_timing',repeat=repeat,frame=index+1,rows=len(rows),seconds=time.monotonic()-started)
                        save(ROOT/'STATUS.json',state);print(json.dumps(state),flush=True)
                observations.append(gpu(f'round_{repeat}_end'))
        assert len(rows)==11880
        save(ROOT/'reports/PEAK_MEMORY.json',dict(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved()))
        event_rows=[call(case,int(f),True) for f in np.linspace(0,791,96,dtype=int) for case in CASES]
        write(ROOT/'reports/CUDA_EVENTS.csv',event_rows)
        evaluate(scene,prepare,cameras,bank,state,started)
    scene.renderer.check_state();ref.check_state()
    after={n:sha(n) for n in before};assert before==after;save(ROOT/'reports/INPUTS_AFTER.json',after)
    assert all(sha(ROOT/'source'/n)==h for n,h in source.items())
    old=json.loads((ROOT/'reports/PRODUCTION_BEFORE.json').read_text());new={n:dict(bytes=Path(n).stat().st_size,sha256=sha(n)) for n in old};assert old==new
    save(ROOT/'reports/PRODUCTION_AFTER.json',new);observations.append(gpu('complete'));save(ROOT/'reports/GPU_OBSERVATIONS.json',observations)
    save(ROOT/'reports/FINAL_INTEGRITY.json',dict(passed=True,protected_production_files=len(old),changed_files=[],source_unchanged=True,checkpoint_unchanged=True,
        timed_calls=11880,event_calls=480,quality_rows=7128,frames=792,training_runs=0,render_images_saved=0,production_updated=False,commit_push=False,
        free_GiB=shutil.disk_usage(ROOT).free/2**30))
    state.update(state='complete',seconds=time.monotonic()-started);save(ROOT/'STATUS.json',state);print('RATIO_SWEEP_COMPLETE',flush=True)


def evaluate(scene,prepare,cameras,bank,state,started):
    from torchmetrics.functional.image import structural_similarity_index_measure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    lpips=LearnedPerceptualImagePatchSimilarity(normalize=True,reduction='none').cuda()
    rows=[];pairs=[];images=[];truths=[];pending=[]
    def flush():
        if not pending:return
        x,y=torch.stack(images),torch.stack(truths);diff=x-y
        values=torch.stack([-10*torch.log10(diff.square().mean((1,2,3))),structural_similarity_index_measure(x,y,data_range=1.,reduction='none'),
            lpips(x,y).reshape(-1),diff.abs().mean((1,2,3))],1);lpips.reset();assert torch.isfinite(values).all()
        for row,v in zip(pending,values.cpu().tolist()):row.update(zip(['psnr','ssim','lpips','photo_l1'],v));rows.append(row)
        images.clear();truths.clear();pending.clear()
    def add(case,backend,frame,image,gt,stats,ids):
        images.append(image['render'].clamp(0,1));truths.append(gt)
        pending.append(dict(case=case,backend=backend,ratio=stats['ratio'],frame=frame,timestamp=scene.views[frame]['timestamp'],
            bank=str(Path(scene.views[frame]['image']).resolve()) in bank,candidate_N=stats['candidate_gaussians'],eligible_N=stats['eligible_candidates'],
            requested_K=stats['requested_budget'],K=len(ids),projected_K=int((image['radii']>0).sum()),shortfall=stats['budget_shortfall']))
        if len(pending)==8:flush()
    for frame in range(792):
        gt=scene.ground_truth(frame);a,ia,sa=prepare('full',frame);image=render_prepared(cameras[frame],a,scene.background,return_radii=True)
        add('full','full',frame,image,gt,sa,ia)
        for case in CASES[:-1]:
            a,ia,sa=prepare(case,frame,'cuda');b,ib,sb=prepare(case,frame)
            assert len(ia)==len(ib) and sa['candidate_gaussians']==sb['candidate_gaussians']
            x=render_prepared(cameras[frame],a,scene.background,return_radii=True);y=render_prepared(cameras[frame],b,scene.background,return_radii=True)
            assert torch.isfinite(y['render']).all();same=torch.equal(ia,ib);diff=(x['render']-y['render']).abs()
            pair=dict(case=case,frame=frame,ids_same=same,id_retention=1. if same or not len(ib) else float(torch.isin(ib,ia).float().mean()),
                image_max_abs=float(diff.max()),image_mean_abs=float(diff.mean()),image_mse=float(diff.square().mean()),radii_same=torch.equal(x['radii'],y['radii']))
            for key in ['means3D','scales','rotations','opacities','dc']:pair[key+'_max_abs']=float((a['inputs'][key]-b['inputs'][key]).abs().max()) if same and len(ia) else None
            pairs.append(pair);add(case,'cuda',frame,x,gt,sa,ia);add(case,'deployment',frame,y,gt,sb,ib)
        if frame%25==0 or frame==791:
            flush();state.update(state='quality',frame=frame+1,quality_rows=len(rows),seconds=time.monotonic()-started);save(ROOT/'STATUS.json',state);print(json.dumps(state),flush=True)
    flush();assert len(rows)==7128 and len(pairs)==3168
    write(ROOT/'evaluation/PER_FRAME.csv',rows);write(ROOT/'evaluation/OUTPUT_DIFFERENCES.csv',pairs)
    lookup={(r['backend'],r['case'],r['frame']):r for r in rows};gates=[]
    for case in CASES[:-1]:
        delta=np.asarray([lookup['deployment',case,f]['psnr']-lookup['cuda',case,f]['psnr'] for f in range(792)])
        gate=dict(case=case,mean_delta=float(delta.mean()),p01_delta=float(np.percentile(delta,1)),worst_delta=float(delta.min()))
        gate['passed']=gate['mean_delta']>=-.01 and gate['p01_delta']>=-.05 and gate['worst_delta']>=-.1;gates.append(gate)
    save(ROOT/'reports/QUALITY_GATES.json',dict(passed=all(v['passed'] for v in gates),cases=gates));assert all(v['passed'] for v in gates)


if __name__=='__main__':
    try:main()
    except Exception as error:
        p=ROOT/'STATUS.json';v=json.loads(p.read_text()) if p.exists() else {};v.update(state='failed',error=repr(error));save(p,v);raise
