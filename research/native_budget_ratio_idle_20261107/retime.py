"""GPU空闲后重测七种局部预算及原版Full；质量沿用相同模型的完整轨迹。"""
from measurement_helpers import *
from budget_viewer import load_views
from gaussian_model import GaussianModel

REQUESTS=REQUESTS+[dict(method='cartgs_full')]


def module(name,path,package=False):
    spec=importlib.util.spec_from_file_location(name,path,submodule_search_locations=[str(Path(path).parent)] if package else None)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value


@torch.no_grad()
def main():
    lock=(ROOT/'reports/run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'STATUS.json').exists()
    state=dict(state='loading',process=identity(),started_unix=time.time());save(ROOT/'STATUS.json',state)
    torch.set_num_threads(2);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    protected=json.loads((ROOT/'reports/INPUTS_BEFORE.json').read_text());assert all(sha(n)==h for n,h in protected.items())
    deployment_runtime.deployment_extension=lambda:load_binary('mycartgs_deployment_fp32_v4')
    frozen_inference_cuda.extension=lambda:load_binary('mycartgs_frozen_inference_v3')
    scene=BudgetScene(CHECKPOINT,DATASET,100,0,None,'pytorch');scene.model.eval()
    runtime=deployment_runtime.DeploymentRenderer.from_checkpoint(CHECKPOINT,0);assert runtime.total==13298 and len(scene.views)==792
    c=scene.calibration
    cameras=[MiniCam(c['width'],c['height'],focal2fov(c['fx'],c['width']),focal2fov(c['fy'],c['height']),np.linalg.inv(np.asarray(v['pose']))) for v in scene.views]
    vanilla=json.loads((ROOT/'reference/PREVIOUS_ENVIRONMENT.json').read_text())['cartgs_history']
    vcal,vviews=load_views(Path(vanilla['checkpoint'])/'ply',DATASET)
    assert vcal==c and len(vviews)==792
    for a,b in zip(vviews,scene.views):assert a['timestamp']==b['timestamp'] and a['image']==b['image']
    vcameras=[MiniCam(c['width'],c['height'],focal2fov(c['fx'],c['width']),focal2fov(c['fy'],c['height']),np.linalg.inv(np.asarray(v['pose']))) for v in vviews]
    pc=GaussianModel(3);pc.load_ply(vanilla['ply_path']);assert len(pc.get_xyz)==35301
    original_raster=module('cartgs_original_raster',ROOT/'source/original_raster/__init__.py',True)
    original_renderer=module('cartgs_original_renderer',ROOT/'source/original_renderer.py')
    original_renderer.GaussianRasterizationSettings=original_raster.GaussianRasterizationSettings
    original_renderer.GaussianRasterizer=original_raster.GaussianRasterizer
    baseline=read(ROOT/'reference/CARTGS_PER_VIEW.csv')
    fixed=read(ROOT/'reference/FIXED_QUALITY.csv');ratio=read(ROOT/'reference/RATIO_PER_VIEW.csv')
    quality={(f'ours_{v["budget"]}',int(v['frame'])):v for v in fixed}
    quality.update({(v['method'],int(v['frame'])):v for v in ratio})
    quality.update({('cartgs_full',int(v['frame'])):v for v in baseline});assert len(quality)==6336
    def call(frame,req,verify=False):
        torch.cuda.synchronize();a=time.perf_counter()
        if req['method']=='cartgs_full':
            b=a;output=original_renderer.render(vcameras[frame],pc,scene.background);ids=None;stats={};K=35301
        else:
            data,ids,stats=runtime.prepare(cameras[frame],**{k:v for k,v in req.items() if k!='method'})
            b=time.perf_counter();output=render_prepared(cameras[frame],data,scene.background);K=len(ids)
        torch.cuda.synchronize();end=time.perf_counter()
        q=quality[req['method'],frame];assert K==int(q['K'])
        if ids is not None:
            target=req.get('budget',math.floor(req.get('ratio_budget',0)*stats['candidate_gaussians']))
            assert K==min(target,stats['eligible_candidates']) and target==stats['requested_budget'] and not stats['cache_hit']
            assert stats['view_conditioned'] and stats['compensation_enabled']
            assert stats['candidate_gaussians']==int(q['candidate_N']) and stats['eligible_candidates']==int(q['eligible_N'])
        if verify:
            image=output['render'].clamp(0,1);psnr=float(-10*torch.log10((image-scene.ground_truth(frame)).square().mean()))
            assert abs(psnr-float(q['psnr']))<1e-5,(req,frame,psnr,q['psnr'])
        return dict(method=req['method'],frame=frame,timestamp=scene.views[frame]['timestamp'],K=K,
            candidate_N=stats.get('candidate_gaussians',35301),eligible_N=stats.get('eligible_candidates',35301),
            requested_K=stats.get('requested_budget',35301),network_ratio=stats.get('ratio',0.),shortfall=stats.get('budget_shortfall',0),
            prepare_ms=(b-a)*1000,render_ms=(end-b)*1000,total_ms=(end-a)*1000)
    save(ROOT/'reports/ENVIRONMENT.json',dict(checkpoint=str(CHECKPOINT),vanilla=vanilla,N=13298,anchors=2790,frames=792,requests=REQUESTS,
        backend=runtime.backend,binary=str(runtime._ext.__file__),binary_sha256=sha(runtime._ext.__file__),torch=torch.__version__,cuda=torch.version.cuda,
        tf32=False,device=torch.cuda.get_device_name(),quality_reused=True,cartgs_timing_fresh=True,images_saved=0,training_runs=0,
        original_extension=str(original_raster._C.__file__),original_extension_sha256=sha(original_raster._C.__file__),threads=2))
    with torch.inference_mode():
        parity=[]
        for frame in (0,396,791):
            for req in REQUESTS:call(frame,req,True);parity.append(dict(frame=frame,method=req['method'],historical_quality_match=True))
        save(ROOT/'reports/QUALITY_PARITY.json',dict(passed=True,cases=parity,tolerance_psnr_dB=1e-5))
        observations=[gpu('before_warmup')]
        for frame in np.linspace(0,791,64,dtype=int):
            for req in REQUESTS:call(int(frame),req)
        monitor=ActivityMonitor();deadline=time.time()+600;accepted=[];attempts=[];all_rows=[]
        torch.cuda.reset_peak_memory_stats()
        try:
            while len(accepted)<3 and time.time()<deadline:
                if not monitor.idle():
                    state.update(state='waiting_quiet_gpu',accepted_rounds=len(accepted),remaining_seconds=round(deadline-time.time()))
                    save(ROOT/'STATUS.json',state);time.sleep(1);continue
                repeat=len(accepted);frames=list(range(792))
                if repeat==1:frames.reverse()
                if repeat==2:random.Random(0).shuffle(frames)
                attempt=len(attempts);start=time.time();values=[];state.update(state='timing',repeat=repeat,attempt=attempt);save(ROOT/'STATUS.json',state)
                for index,frame in enumerate(frames):
                    shift=(index+repeat)%len(REQUESTS)
                    for req in REQUESTS[shift:]+REQUESTS[:shift]:
                        row=call(frame,req);row.update(repeat=repeat,order=index,attempt=attempt);values.append(row)
                end=time.time();time.sleep(4);decision=monitor.assess(start,end)
                item=dict(attempt=attempt,repeat=repeat,start_unix=start,end_unix=end,rows=len(values),**decision);attempts.append(item)
                for row in values:row['accepted']=decision['accepted']
                all_rows.extend(values);write(ROOT/'reports/ALL_TIMINGS.csv',all_rows);save(ROOT/'reports/TIMING_ATTEMPTS.json',attempts)
                if decision['accepted']:accepted.append(values)
                observations.append(gpu(f'after_attempt_{attempt}'))
                print('ROUND_DECISION',json.dumps({k:v for k,v in item.items() if k!='samples'}),flush=True)
        finally:monitor.close()
        times=[row for values in accepted for row in values]
        if times:write(ROOT/'reports/TIMINGS.csv',times)
        save(ROOT/'reports/TIMING_ACCEPTANCE.json',dict(complete=len(accepted)==3,accepted_rounds=len(accepted),accepted_rows=len(times),expected_rows=19008,
            attempts=len(attempts),criterion='pmon external activity; entire rounds; no latency-based filtering',resolution_seconds=1))
        save(ROOT/'reports/PEAK_MEMORY.json',dict(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved()))
    observations.append(gpu('complete'));save(ROOT/'reports/GPU_OBSERVATIONS.json',observations)
    assert all(sha(n)==h for n,h in protected.items());save(ROOT/'reports/INPUTS_AFTER.json',{n:sha(n) for n in protected})
    before=json.loads((ROOT/'reports/PRODUCTION_BEFORE.json').read_text());after={n:dict(bytes=Path(n).stat().st_size,sha256=sha(n)) for n in before}
    assert before==after;save(ROOT/'reports/PRODUCTION_AFTER.json',after)
    save(ROOT/'reports/FINAL_INTEGRITY.json',dict(passed=True,production_files=len(before),input_unchanged=True,images_saved=0,training_runs=0,production_updated=False,commit_push=False))
    state.update(state='complete' if len(accepted)==3 else 'timing_blocked',accepted_rounds=len(accepted),completed_unix=time.time());save(ROOT/'STATUS.json',state);print(json.dumps(state),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        p=ROOT/'STATUS.json';s=json.loads(p.read_text()) if p.exists() else {};s.update(state='failed',error=repr(e));save(p,s);raise
