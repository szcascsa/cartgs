"""相同新G/R部署流程，配对Teacher、Student和直接Master Full。"""
from pathlib import Path
import sys
REFERENCE=Path('/home/prmi/zwj/experiments/sparse_rasterizer_sixhour_20261110_0053')
sys.path.insert(0,str(REFERENCE))
from harness import *
from frozen_renderer import create_renderer,create_camera_preparer
from compensation_student import load_student
from deployment_runtime import DeploymentRenderer

spec=importlib.util.spec_from_file_location('teacher_benefit_helpers',
    '/home/prmi/zwj/experiments/sparse_teacher_lod_benefit_20261110/benchmark.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
OUT=Path(__file__).resolve().parent
MODEL_ROOT=Path('/home/prmi/zwj/experiments/compensator_distillation_replica_all_20261108/models')
SLUGS=['Replica__office1__mono','Replica__office2__rgbd','Replica__room0__rgbd','Replica__room2__mono']
RATIOS=[.01,.05,.1,.3,.7]
CONDITIONS=['full']+[f'{v}_{r}' for r in RATIOS for v in ['teacher','student']]


@torch.no_grad()
def main():
    reports=OUT/'reports';reports.mkdir(exist_ok=False)
    protected=json.loads((Path('/home/prmi/zwj/experiments/sparse_teacher_lod_benefit_20261110')/'reports/PROTECTED_AFTER.json').read_text())
    protected={p:helpers.sha_file(p) for p in protected}
    for slug in SLUGS:
        p=MODEL_ROOT/slug/'continuous_calibration/best.pt';protected[str(p)]=helpers.sha_file(p)
    write_json(reports/'PROTECTED_BEFORE.json',protected)
    write_json(reports/'PROTOCOL.json',dict(slugs=SLUGS,ratios=RATIOS,rounds=2,repeats=6,views=18,
        scopes=['raster','prepare','pipeline'],renderer='frozen G',camera='frozen R',
        full='direct frozen Master packet; no per-frame Full clone',student_stage='continuous_calibration/best.pt',
        teacher_mlp=25742,student_mlp=3534,normal_training_unchanged=True,
        fr1_xyz_excluded='existing student fingerprint differs from prior benchmark checkpoint',
        evaluation_gt_outside_timing=True))
    assert not external_load()['active'],'GPU busy'
    monitor=helpers.Monitor();samples=[];checks=[];quality=[];identities=[]
    for slug in SLUGS:
        scene,source=load_scene(slug)
        teacher=source.runtime;assert teacher.student is None
        student_path=MODEL_ROOT/slug/'continuous_calibration/best.pt'
        student,payload=load_student(student_path,'cuda',source.fingerprint)
        student_engine=DeploymentRenderer(source.scene.model,source.scene.gaussians,source.scene.config,student=student)
        assert sum(p.numel() for p in student.parameters())==3534
        assert sum(p.numel() for p in teacher._base.spatial.parameters())==25742
        identities.append(dict(scene=scene['id'],N=teacher.total,checkpoint=scene['checkpoint'],
            student=str(student_path),student_sha256=helpers.sha_file(student_path),fingerprint=source.fingerprint,
            student_step=payload['step'],teacher_params=25742,student_params=3534))
        factory=create_camera_preparer(MiniCam,variant='R');views=views_for(source,'final')
        camera=factory.from_source(source,views[0][1])
        renderer=create_renderer(camera,source.scene.background,teacher.total,use_graph=True)
        master,master_ids=teacher.full()
        def packet_for(cam,condition):
            if condition=='full':return master,master_ids,{}
            version,ratio=condition.split('_')
            return (teacher if version=='teacher' else student_engine).prepare(cam,ratio_budget=float(ratio))
        for _,frame in views:
            cam=factory.from_source(source,frame);original_cam=source.camera(frame)
            assert all(bits_equal(getattr(cam,k),getattr(original_cam,k)) for k in
                ['world_view_transform','projection_matrix','full_proj_transform','camera_center'])
            gt=source.scene.ground_truth(frame).clamp(0,1)
            for condition in CONDITIONS:
                packet,ids,_=packet_for(cam,condition)
                if condition.startswith('student'):
                    _,teacher_ids,_=packet_for(cam,condition.replace('student','teacher'))
                    assert torch.equal(ids,teacher_ids)
                ref=render_prepared(cam,packet,source.scene.background,return_radii=True)
                image,radii=renderer.render(cam,packet,source.scene.background)
                assert bits_equal(image,ref['render']) and bits_equal(radii,ref['radii']),(slug,condition,frame)
                checks.append(dict(scene=scene['id'],frame=frame,condition=condition,K=len(ids),
                    ids_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),rgb_radii_bitwise=True))
                psnr=float(-10*torch.log10((image.clamp(0,1)-gt).square().mean().clamp_min(1e-12)))
                quality.append(dict(scene=scene['id'],frame=frame,condition=condition,K=len(ids),psnr=psnr))
        for condition in CONDITIONS:
            for _,frame in views:
                cam=factory.from_source(source,frame);packet,ids,_=packet_for(cam,condition)
                renderer.render(cam,packet,source.scene.background)
        torch.cuda.synchronize();monitor.check()
        for round_index in range(2):
            ordered=views if round_index==0 else list(reversed(views))
            for repeat in range(6):
                for index,(_,frame) in enumerate(ordered):
                    cam=factory.from_source(source,frame)
                    packets={condition:packet_for(cam,condition) for condition in CONDITIONS}
                    for ratio in RATIOS:
                        assert torch.equal(packets[f'teacher_{ratio}'][1],packets[f'student_{ratio}'][1])
                    for scope in ['raster','prepare','pipeline']:
                        for condition in helpers.rotate(CONDITIONS,index+repeat+round_index*3):
                            torch.cuda.synchronize();tick=time.perf_counter_ns()
                            if scope=='raster':current=cam;packet,ids,stats=packets[condition]
                            else:
                                current=factory.from_source(source,frame);packet,ids,stats=packet_for(current,condition)
                            if scope!='prepare':image,radii=renderer.render(current,packet,source.scene.background)
                            torch.cuda.synchronize();ms=(time.perf_counter_ns()-tick)/1e6
                            samples.append(dict(scene=scene['id'],condition=condition,scope=scope,round=round_index,
                                repeat=repeat,frame=frame,N=teacher.total,K=len(ids),wall_ms=ms))
                    del packets
                monitor.check();print('STUDENT_PAIRED_PROGRESS',scene['id'],round_index,repeat,flush=True)
        for name,rows in [('samples',samples),('correctness',checks),('quality',quality)]:write_csv(reports/(name+'.csv'),rows)
        assert renderer.native.stats()[1:3]==[0,0]
        del source,teacher,student,student_engine,factory,renderer,master,master_ids,packet,ids,ref,image,radii,gt
        torch.cuda.empty_cache()
    monitor.close();write_json(reports/'GPU_OBSERVATIONS.json',monitor.rows)
    write_json(reports/'MODEL_IDENTITIES.json',identities)
    after={p:helpers.sha_file(p) for p in protected};changed=[p for p in protected if protected[p]!=after[p]]
    write_json(reports/'PROTECTED_AFTER.json',after)
    result=dict(passed=not changed,changed=changed,protected_files=len(protected),scenes=len(identities),
        samples=len(samples),correctness_cases=len(checks),quality_rows=len(quality),
        external_active_samples=0,selection_ids_identical=True,student_binding_verified=True,
        renderer='frozen G',camera='frozen R',training=False,production_changed=False,saved_images=0)
    write_json(reports/'VALIDATION.json',result);assert result['passed']
    print('STUDENT_PAIRED_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
