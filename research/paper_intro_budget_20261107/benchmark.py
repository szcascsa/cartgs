"""五种本方法完整评价及配对计时；原版仅作环境控制。"""
from measurement_helpers import *
from budget_viewer import load_views
from gaussian_model import GaussianModel

REQUESTS=[dict(method='fixed_2000',budget=2000),dict(method='fixed_4000',budget=4000),
    dict(method='ratio_50',ratio_budget=.5),dict(method='ratio_80',ratio_budget=.8),dict(method='ours_full'),dict(method='cartgs_control')]


def module(name,path,package=False):
    spec=importlib.util.spec_from_file_location(name,path,submodule_search_locations=[str(Path(path).parent)] if package else None)
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj);return obj


@torch.no_grad()
def main():
    lock=(ROOT/'reports/run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (ROOT/'STATUS.json').exists()
    state=dict(state='loading',process=identity(),started_unix=time.time());save(ROOT/'STATUS.json',state)
    torch.set_num_threads(2);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    protected=json.loads((ROOT/'reports/INPUTS_BEFORE.json').read_text());assert all(sha(n)==h for n,h in protected.items())
    deployment_runtime.deployment_extension=lambda:load_binary('mycartgs_deployment_fp32_v4')
    frozen_inference_cuda.extension=lambda:load_binary('mycartgs_frozen_inference_v3')
    scene=BudgetScene(CHECKPOINT,DATASET,2000,0,None,'pytorch');scene.model.eval()
    runtime=deployment_runtime.DeploymentRenderer.from_checkpoint(CHECKPOINT,0);assert runtime.total==13298 and len(scene.views)==792
    c=scene.calibration
    cameras=[MiniCam(c['width'],c['height'],focal2fov(c['fx'],c['width']),focal2fov(c['fy'],c['height']),np.linalg.inv(np.asarray(v['pose']))) for v in scene.views]
    vanilla=json.loads((ROOT/'reference/PREVIOUS_ENVIRONMENT.json').read_text())['vanilla']
    vcal,vviews=load_views(Path(vanilla['checkpoint'])/'ply',DATASET);assert vcal==c and len(vviews)==792
    assert all(a['timestamp']==b['timestamp'] and a['image']==b['image'] for a,b in zip(vviews,scene.views))
    vcameras=[MiniCam(c['width'],c['height'],focal2fov(c['fx'],c['width']),focal2fov(c['fy'],c['height']),np.linalg.inv(np.asarray(v['pose']))) for v in vviews]
    pc=GaussianModel(3);pc.load_ply(vanilla['ply_path']);assert len(pc.get_xyz)==35301
    original_raster=module('cartgs_original_raster',ROOT/'source/original_raster/__init__.py',True)
    original_renderer=module('cartgs_original_renderer',ROOT/'source/original_renderer.py')
    original_renderer.GaussianRasterizationSettings=original_raster.GaussianRasterizationSettings;original_renderer.GaussianRasterizer=original_raster.GaussianRasterizer
    path=Path('/home/prmi/zwj/experiments/fr1xyz_viewer_budget_comparison_20261105/build/fr1xyz_production_counts.so')
    binding=module('fr1xyz_production_counts',path)
    history=read(ROOT/'reference/PREVIOUS_PER_VIEW.csv');cart={int(v['frame']):v for v in history if v['method']=='cartgs_full'};assert len(cart)==792
    def prepare(frame,req):
        if req['method']=='ours_full':
            data,ids=runtime.full();stats=dict(candidate_gaussians=13298,eligible_candidates=13298,ratio=None,requested_budget=13298,budget_shortfall=0,cache_hit=False)
        else:data,ids,stats=runtime.prepare(cameras[frame],**{k:v for k,v in req.items() if k!='method'})
        if req['method']!='ours_full':
            target=req.get('budget',math.floor(req.get('ratio_budget',0)*stats['candidate_gaussians']))
            assert len(ids)==min(target,stats['eligible_candidates']) and target==stats['requested_budget'] and not stats['cache_hit']
        else:assert len(ids)==13298
        return data,ids,stats
    def call(frame,req):
        torch.cuda.synchronize();a=time.perf_counter()
        if req['method']=='cartgs_control':
            b=a;out=original_renderer.render(vcameras[frame],pc,scene.background);K=35301;stats={}
        else:
            data,ids,stats=prepare(frame,req);b=time.perf_counter();out=render_prepared(cameras[frame],data,scene.background);K=len(ids)
        torch.cuda.synchronize();end=time.perf_counter()
        return dict(method=req['method'],frame=frame,timestamp=scene.views[frame]['timestamp'],K=K,
            candidate_N=stats.get('candidate_gaussians',35301),eligible_N=stats.get('eligible_candidates',35301),network_ratio=stats.get('ratio'),
            requested_K=stats.get('requested_budget',35301),shortfall=stats.get('budget_shortfall',0),prepare_ms=(b-a)*1000,render_ms=(end-b)*1000,total_ms=(end-a)*1000),out
    save(ROOT/'reports/ENVIRONMENT.json',dict(checkpoint=str(CHECKPOINT),vanilla=vanilla,N=13298,anchors=2790,frames=792,requests=REQUESTS,
        backend=runtime.backend,binary=str(runtime._ext.__file__),binary_sha256=sha(runtime._ext.__file__),torch=torch.__version__,cuda=torch.version.cuda,
        tf32=False,device=torch.cuda.get_device_name(),full_path='runtime.full() per request + render_prepared; full copy included',
        original_extension=str(original_raster._C.__file__),cartgs_plot_reused=True,baseline_control_drift_threshold_percent=5,training_runs=0,render_images_saved=0))
    with torch.inference_mode():
        parity=[]
        for frame in (0,396,791):
            for req in REQUESTS[:-1]:
                data,ids,stats=prepare(frame,req);normal=render_prepared(cameras[frame],data,scene.background,return_radii=True);out=counted(binding,cameras[frame],data,scene.background)
                assert bitwise(normal['render'],out[2]) and torch.equal(normal['radii'],out[3]);parity.append(dict(frame=frame,method=req['method'],color_and_radii_bitwise=True))
            rec,out=call(frame,REQUESTS[-1]);psnr=float(-10*torch.log10((out['render'].clamp(0,1)-scene.ground_truth(frame)).square().mean()))
            assert abs(psnr-float(cart[frame]['psnr']))<1e-5
        save(ROOT/'reports/RENDERER_PARITY.json',dict(passed=True,cases=parity,cartgs_representative_quality_checks=3))
        quality=[]
        for frame,v in enumerate(scene.views):
            gt=scene.ground_truth(frame)
            for req in REQUESTS[:-1]:
                data,ids,stats=prepare(frame,req);out=counted(binding,cameras[frame],data,scene.background);image=out[2].clamp(0,1)
                psnr=float(-10*torch.log10((image-gt).square().mean()));assert math.isfinite(psnr) and torch.isfinite(out[2]).all()
                row=dict(method=req['method'],frame=frame,timestamp=v['timestamp'],image=v['image'],map_N=13298,K=len(ids),candidate_N=stats['candidate_gaussians'],
                    eligible_N=stats['eligible_candidates'],requested_K=stats['requested_budget'],network_ratio=stats['ratio'],shortfall=stats['budget_shortfall'],
                    projected_K=int((out[3]>0).sum()),contributing_K=int((out[5]>0).sum()),psnr=psnr,photo_l1=float((image-gt).abs().mean()))
                assert 0<=row['contributing_K']<=row['projected_K']<=row['K'];quality.append(row)
            if frame%100==0 or frame==791:
                state.update(state='quality',frames=frame+1,rows=len(quality));save(ROOT/'STATUS.json',state);print(json.dumps(state),flush=True)
        write(ROOT/'evaluation/OURS_PER_VIEW.csv',quality);assert len(quality)==3960
        observations=[gpu('before_warmup')]
        for frame in np.linspace(0,791,64,dtype=int):
            for req in REQUESTS:call(int(frame),req)
        monitor=ActivityMonitor();deadline=time.time()+600;accepted=[];attempts=[];all_rows=[]
        try:
            while len(accepted)<3 and time.time()<deadline:
                if not monitor.idle():
                    state.update(state='waiting_quiet_gpu',accepted_rounds=len(accepted),remaining_seconds=round(deadline-time.time()));save(ROOT/'STATUS.json',state);time.sleep(1);continue
                repeat=len(accepted);frames=list(range(792))
                if repeat==1:frames.reverse()
                if repeat==2:random.Random(0).shuffle(frames)
                start=time.time();values=[];attempt=len(attempts);state.update(state='timing',repeat=repeat,attempt=attempt);save(ROOT/'STATUS.json',state)
                for index,frame in enumerate(frames):
                    shift=(index+repeat)%len(REQUESTS)
                    for req in REQUESTS[shift:]+REQUESTS[:shift]:
                        row,out=call(frame,req);row.update(repeat=repeat,order=index,attempt=attempt);values.append(row)
                end=time.time();time.sleep(4);decision=monitor.assess(start,end);item=dict(attempt=attempt,repeat=repeat,start_unix=start,end_unix=end,rows=len(values),**decision);attempts.append(item)
                for row in values:row['accepted']=decision['accepted']
                all_rows.extend(values);write(ROOT/'reports/ALL_TIMINGS.csv',all_rows);save(ROOT/'reports/TIMING_ATTEMPTS.json',attempts)
                if decision['accepted']:accepted.append(values)
                observations.append(gpu(f'after_attempt_{attempt}'));print('ROUND_DECISION',json.dumps({k:v for k,v in item.items() if k!='samples'}),flush=True)
        finally:monitor.close()
        times=[row for values in accepted for row in values]
        if times:write(ROOT/'reports/TIMINGS.csv',times)
        save(ROOT/'reports/TIMING_ACCEPTANCE.json',dict(complete=len(accepted)==3,accepted_rounds=len(accepted),accepted_rows=len(times),expected_rows=14256,attempts=len(attempts),resolution_seconds=1,latency_based_filtering=False))
    observations.append(gpu('complete'));save(ROOT/'reports/GPU_OBSERVATIONS.json',observations)
    assert all(sha(n)==h for n,h in protected.items());save(ROOT/'reports/INPUTS_AFTER.json',{n:sha(n) for n in protected})
    before=json.loads((ROOT/'reports/PRODUCTION_BEFORE.json').read_text());after={n:dict(bytes=Path(n).stat().st_size,sha256=sha(n)) for n in before};assert before==after;save(ROOT/'reports/PRODUCTION_AFTER.json',after)
    save(ROOT/'reports/FINAL_INTEGRITY.json',dict(passed=True,production_files=len(before),inputs_unchanged=True,training_runs=0,render_images_saved=0,production_updated=False,commit_push=False))
    state.update(state='complete' if len(accepted)==3 else 'timing_blocked',accepted_rounds=len(accepted),completed_unix=time.time());save(ROOT/'STATUS.json',state);print(json.dumps(state),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        p=ROOT/'STATUS.json';s=json.loads(p.read_text()) if p.exists() else {};s.update(state='failed',error=repr(e));save(p,s);raise
