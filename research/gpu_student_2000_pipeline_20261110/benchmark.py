"""GPU Student在1200/2000宽度下的真实准备、渲染与完整推理。"""
from pathlib import Path
import sys,importlib.util,json,time,math,gc,hashlib
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PRIOR=Path('/home/prmi/zwj/experiments/sparse_student_lod_benefit_20261110')
spec=importlib.util.spec_from_file_location('gpu_student_helpers',PRIOR/'benchmark.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
torch,np=h.torch,h.np
SIZES=[(1200,680),(2000,1133)]
RATIOS=h.RATIOS
CONDITIONS=['full']+[f'local{r:g}' for r in RATIOS]
SHA=h.helpers.sha_file


def configure():
    import deployment_runtime,frozen_inference_cuda
    expected={
        'deployment_runtime':('/home/prmi/zwj/experiments/compensator_distillation_20261108/build/mycartgs_deployment_fp32_v31/mycartgs_deployment_fp32_v31.so','fa7b855685ecd4f75ce7118ee07fadd27640af08099a6775ffee5fe87fa7d601'),
        'frozen_inference_cuda':('/home/prmi/zwj/experiments/deployment_device_counts_20261107/build/mycartgs_frozen_inference_v3/mycartgs_frozen_inference_v3.so','d44bdfb113314d96a3b38e63e77ba4678f6070a9c0ece1a4f1fb3005b8ef85fb')}
    for name,module in [('deployment_runtime',deployment_runtime),('frozen_inference_cuda',frozen_inference_cuda)]:
        path,sha=expected[name];assert SHA(path)==sha
        native=(module.deployment_extension if name=='deployment_runtime' else module.extension)()
        assert Path(native.__file__).resolve()==Path(path).resolve()
    return expected


@torch.no_grad()
def main():
    reports=HERE/'reports';reports.mkdir(exist_ok=False)
    binaries=configure()
    torch.set_num_threads(2)
    protected=json.loads((Path('/home/prmi/zwj/experiments/render_resolution_2000_20261110')/'reports/PROTECTED_AFTER.json').read_text())
    for path,sha in binaries.values():protected[path]=sha
    protected={path:SHA(path) for path in protected}
    h.write_json(reports/'PROTECTED_BEFORE.json',protected)
    h.write_json(reports/'PROTOCOL.json',dict(slugs=h.SLUGS,ratios=RATIOS,sizes=SIZES,rounds=2,repeats=6,views=18,
        renderer='frozen G',camera='frozen R',student='32/32 four direct Heads; 3534 parameters',
        device='Selector and Student both GPU; CPU handles camera inverse and submission',
        camera_control='same pose/FoV/projection; actual width/height passed into view features, no frozen predictions',
        full='direct fixed Master attributes; current R camera every request',
        scopes=['prepare_wall','raster_wall','raster_event','pipeline_wall'],
        prepare='camera CPU inverse + R GPU camera + candidates/Selector/Compensator/pack + final sync; no rendering',
        pipeline='current camera + GPU candidate/selection/student/pack + G rendering + final sync, no input/result cache',
        event='render-only CUDA Event with already prepared current camera and attributes; no CPU inference or data transfer',
        scheduling='two resolutions paired and rotated; reverse scene view/condition order in second round',
        numerical_validation='same-resolution A/G RGB/radii bitwise; original-size camera tensors vs prior API bitwise',
        quality_validation=False,training=False,production_changed=False,saved_images=False))
    samples=[];checks=[];identities=[];cross=[]
    for slug in h.SLUGS:
        scene,source=h.load_scene(slug)
        student_path=h.MODEL_ROOT/slug/'continuous_calibration/best.pt'
        student,payload=h.load_student(student_path,'cuda',source.fingerprint)
        engine=h.DeploymentRenderer(source.scene.model,source.scene.gaussians,source.scene.config,student=student)
        assert sum(p.numel() for p in student.parameters())==3534
        calibration=source.scene.calibration
        assert (calibration['width'],calibration['height'])==SIZES[0]
        fx=2*math.atan(calibration['width']/(2*calibration['fx']))
        fy=2*math.atan(calibration['height']/(2*calibration['fy']))
        factories={size:h.create_camera_preparer(h.MiniCam,variant='R') for size in SIZES}
        views=h.views_for(source,'final');renderers={}
        def camera(size,frame):
            return factories[size].create(size[0],size[1],fx,fy,np.linalg.inv(np.asarray(source.scene.views[frame]['pose'])))
        for size in SIZES:
            renderers[size]=h.create_renderer(camera(size,views[0][1]),source.scene.background,engine.total,use_graph=True)
        master,master_ids=engine.full()
        def packet_for(cam,condition):
            if condition=='full':return master,master_ids,{}
            return engine.prepare(cam,ratio_budget=float(condition[5:]))
        identities.append(dict(scene=scene['id'],N=engine.total,checkpoint=scene['checkpoint'],fingerprint=source.fingerprint,
            student=str(student_path),student_sha256=SHA(student_path),student_step=payload['step'],parameters=3534))
        for _,frame in views:
            saved_ids={}
            for size in SIZES:
                cam=camera(size,frame)
                if size==SIZES[0]:
                    original=source.camera(frame)
                    assert all(h.bits_equal(getattr(cam,key),getattr(original,key)) for key in
                        ['world_view_transform','projection_matrix','full_proj_transform','camera_center'])
                for condition in CONDITIONS:
                    packet,ids,stats=packet_for(cam,condition)
                    ref=h.render_prepared(cam,packet,source.scene.background,return_radii=True)
                    image,radii=renderers[size].render(cam,packet,source.scene.background)
                    assert h.bits_equal(image,ref['render']) and h.bits_equal(radii,ref['radii']),(slug,frame,size,condition)
                    assert torch.isfinite(image).all()
                    checks.append(dict(scene=scene['id'],frame=frame,width=size[0],height=size[1],condition=condition,
                        N=engine.total,M=stats.get('candidate_gaussians',engine.total),K=len(ids),
                        ids_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),rgb_radii_bitwise=True))
                    if size==SIZES[0]:saved_ids[condition]=ids.clone()
                    else:
                        low=saved_ids[condition];overlap=int(torch.isin(ids,low).sum())
                        cross.append(dict(scene=scene['id'],frame=frame,condition=condition,K1200=len(low),K2000=len(ids),
                            ids_same=torch.equal(ids,low),intersection=overlap,union=len(low)+len(ids)-overlap))
        for _ in range(2):
            for _,frame in views:
                for size in SIZES:
                    cam=camera(size,frame)
                    for condition in CONDITIONS:
                        packet,ids,stats=packet_for(cam,condition);renderers[size].render(cam,packet,source.scene.background)
        torch.cuda.synchronize()
        for round_index in range(2):
            ordered=views if round_index==0 else list(reversed(views))
            for repeat in range(6):
                for vi,(_,frame) in enumerate(ordered):
                    for ci,condition in enumerate(h.helpers.rotate(CONDITIONS,repeat+vi+round_index)):
                        sizes=SIZES if (repeat+vi+ci+round_index)%2==0 else list(reversed(SIZES))
                        for size in sizes:
                            cam=camera(size,frame);packet,ids,stats=packet_for(cam,condition)
                            for scope in ['prepare_wall','raster_wall','pipeline_wall']:
                                torch.cuda.synchronize();tick=time.perf_counter_ns()
                                if scope!='raster_wall':
                                    current=camera(size,frame);p,i,st=packet_for(current,condition)
                                else:current,p,i,st=cam,packet,ids,stats
                                if scope!='prepare_wall':image,radii=renderers[size].render(current,p,source.scene.background)
                                torch.cuda.synchronize();elapsed=(time.perf_counter_ns()-tick)/1e6
                                samples.append(dict(scene=scene['id'],round=round_index,repeat=repeat,frame=frame,
                                    width=size[0],height=size[1],condition=condition,scope=scope,N=engine.total,
                                    M=st.get('candidate_gaussians',engine.total),K=len(i),ms=elapsed))
                            if repeat<2:
                                torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                                start.record();renderers[size].render(cam,packet,source.scene.background);end.record();end.synchronize()
                                samples.append(dict(scene=scene['id'],round=round_index,repeat=repeat,frame=frame,
                                    width=size[0],height=size[1],condition=condition,scope='raster_event',N=engine.total,
                                    M=stats.get('candidate_gaussians',engine.total),K=len(ids),ms=start.elapsed_time(end)))
                print('PROGRESS',scene['id'],round_index,repeat,len(samples),flush=True)
        for renderer in renderers.values():assert renderer.native.stats()[1:3]==[0,0]
        for name,rows in [('samples',samples),('checks',checks),('cross_resolution_ids',cross)]:h.write_csv(reports/(name+'.csv'),rows)
        h.write_json(reports/'MODEL_IDENTITIES.json',identities)
        del source,student,engine,factories,renderers,renderer,master,master_ids,packet,ids,ref,image,radii,p,i,cam,current,saved_ids
        gc.collect();torch.cuda.empty_cache()
    after={path:SHA(path) for path in protected};changed=[path for path in protected if protected[path]!=after[path]]
    h.write_json(reports/'PROTECTED_AFTER.json',after)
    result=dict(passed=not changed,changed=changed,protected_files=len(protected),samples=len(samples),checks=len(checks),
        scenes=len(identities),cross_resolution_ids_differ=sum(not row['ids_same'] for row in cross),
        training=False,production_changed=False,saved_images=0,gpu_student=True)
    h.write_json(reports/'VALIDATION.json',result);assert result['passed'];print('COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
