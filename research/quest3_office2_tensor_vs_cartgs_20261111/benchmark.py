"""RTX3090模拟Quest3双次2064×2208；仅CaRtGS Vanilla与Tensor＋新准备D。"""
from pathlib import Path
import importlib.util,sys,json,time,gc,hashlib
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PRIOR=Path('/home/prmi/zwj/experiments/tensor_core_native1200_speed_20261111')
spec=importlib.util.spec_from_file_location('quest_tensor_helpers',PRIOR/'helpers.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
p,torch,np=h.p,h.torch,h.np
from scipy.spatial.transform import Rotation
from gaussian_model import GaussianModel
import renderer,diff_gaussian_rasterization
SIZE=(2064,2208)
SLUG='Replica__office2__rgbd'
VANILLA=Path('/home/prmi/zwj/realLOD/vanilla/vanilla_all_datasets_20261031_2011/jobs/replica/rgbd/office2/repeat3/seed0/attempt_1')
ORIGINAL_SHA='27b5b468cf385dd1c96aeabdd65426da38a938a38d52d8181b1b1da485090c86'
CONDITIONS=['cartgs_full','ours_full']+[f'ours_{r:g}' for r in h.RATIOS]


def save(name,value):p.save(HERE/'reports'/name,value)
def write(name,value):p.write_csv(HERE/'reports'/name,value)


@torch.no_grad()
def main():
    reports=HERE/'reports';reports.mkdir(exist_ok=False)
    c,d,cm,dm=h.configure();torch.set_num_threads(2)
    assert h.sha(diff_gaussian_rasterization._C.__file__)==ORIGINAL_SHA
    before=json.loads((PRIOR/'reports/PROTECTED_AFTER.json').read_text())
    cp=VANILLA/'model/15900_shutdown/ply'
    ply,=list((cp/'point_cloud').glob('iteration_*/point_cloud.ply'))
    for path in [ply,cp/'cameras.json',VANILLA/'model/CameraTrajectory_TUM.txt',VANILLA.parents[6]/'reports/PER_RUN.csv']:
        if path.is_file():before[str(path)]=h.sha(path)
    before={name:h.sha(name) for name in before};save('PROTECTED_BEFORE.json',before)
    scene,source=p.load_scene(SLUG);student,student_path,payload=p.student_for(source,SLUG)
    engine=h.engine(source,student,'D');assert engine.total==43197
    camera=h.Camera(source,'D',dm,SIZE)
    original_pc=GaussianModel(3);original_pc.load_ply(str(ply),'cuda')
    for name in ['_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest']:getattr(original_pc,name).requires_grad_(False)
    assert len(original_pc.get_xyz)==84203
    records=np.atleast_2d(np.loadtxt(VANILLA/'model/CameraTrajectory_TUM.txt'))
    timestamps={float(row[0]):row for row in records}
    calibration=json.loads((cp/'cameras.json').read_text())[0]
    assert all(calibration[key]==source.scene.calibration[key] for key in ['width','height','fx','fy'])
    fovx=camera.fx;fovy=camera.fy
    views=h.frames_for(source)
    vanilla_poses={}
    for motion,frame in views:
        view=source.scene.views[frame];stamp=float(view['timestamp']);row=timestamps[stamp]
        pose=np.eye(4);pose[:3,:3]=Rotation.from_quat(row[4:8]).as_matrix();pose[:3,3]=row[1:4]
        vanilla_poses[frame]=pose
        assert int(round(stamp))==int(Path(view['image']).stem.removeprefix('frame'))
    master,master_ids=engine.full()
    render=p.create_renderer(camera.create(views[0][1]),source.scene.background,engine.total,use_graph=True)
    save('PROTOCOL.json',dict(dataset='Replica/office2/rgbd',width=SIZE[0],height=SIZE[1],eyes=2,
        pixels_per_eye=SIZE[0]*SIZE[1],pixels_per_display_frame=2*SIZE[0]*SIZE[1],
        views=18,rounds=3,repeats=8,ratios=h.RATIOS,conditions=CONDITIONS,
        simulation='same current monocular camera twice, two independent prepare+render calls; no IPD or eye offset; not Quest hardware',
        execution='same GPU stream, two serial independent eye renders; no multiview, no reused mask/prediction; hold both outputs',
        field_of_view='original Replica horizontal/vertical FoV; not Quest lens calibration or reprojection',
        maps='separate learned maps, same RGB timestamps and each own SLAM trajectory; Vanilla N84203, ours N43197',
        vanilla='archived highest PSNR repeat3; project vanilla algorithm with extensions disabled, original frozen rasterizer inference path',
        ours='only D Tensor Core+new preparation+native camera+G renderer; same 3534-param student',
        prepare='two current cameras plus attributes; ours LoD includes candidate/selector/student/selection/decode; original Full activates attributes per eye',
        raster='two rasterizer calls only, prepared camera+attributes, CUDA Events per-eye and paired; no network inference',
        full='ours uses frozen resident Master attributes; original activates Full attributes each eye (activation belongs to preparation)',
        scopes=['prepare_pair_wall','raster_pair_event','raster_eye0_event','raster_eye1_event','raster_pair_wall','pipeline_pair_wall'],
        timing='separate measurements, no summing independent medians; render Event includes rasterizer-internal launch/dependency gaps',
        excluded='load, graph precompile, GT, image save/display, tracking/mapping, OpenXR, distortion/timewarp, foveation',
        training=False,production_changed=False,images_saved=False))
    save('MODEL_IDENTITIES.json',dict(ours=scene,ours_N=engine.total,student=str(student_path),student_sha256=h.sha(student_path),
        student_step=payload['step'],teacher_fingerprint=source.fingerprint,student_parameters=3534,
        vanilla_checkpoint=str(cp),vanilla_ply=str(ply),vanilla_N=len(original_pc.get_xyz),vanilla_ply_sha256=h.sha(ply),
        original_renderer=str(renderer.__file__),original_renderer_sha256=h.sha(renderer.__file__),
        original_rasterizer=str(diff_gaussian_rasterization._C.__file__),original_rasterizer_sha256=ORIGINAL_SHA,
        D_binary=d['binary'],D_sha256=d['binary_sha256'],views=[dict(motion=motion,frame=frame,timestamp=source.scene.views[frame]['timestamp']) for motion,frame in views]))
    def prepare(frame,condition):
        if condition=='cartgs_full':
            cam=p.MiniCam(*SIZE,fovx,fovy,np.linalg.inv(vanilla_poses[frame]))
            packet=renderer.prepare_render_data(original_pc)
            return cam,packet,None,dict(K=len(original_pc.get_xyz),M=len(original_pc.get_xyz),N=len(original_pc.get_xyz))
        cam=camera.create(frame)
        if condition=='ours_full':packet,ids,stats=master,master_ids,dict(candidate_gaussians=engine.total)
        else:packet,ids,stats=engine.prepare(cam,ratio_budget=float(condition[5:]))
        return cam,packet,ids,dict(K=len(ids),M=stats.get('candidate_gaussians',len(ids)),N=engine.total)
    def raster(cam,packet,condition):
        if condition=='cartgs_full':
            out=p.render_prepared(cam,packet,source.scene.background,return_radii=True)
            return out['render'],out['radii']
        return render.render(cam,packet,source.scene.background)
    checks=[];rows=[]
    for _,frame in views:
        for condition in CONDITIONS:
            eyes=[prepare(frame,condition) for _ in range(2)];outputs=[]
            for eye,(cam,packet,ids,stats) in enumerate(eyes):
                image,radii=raster(cam,packet,condition);ref=p.render_prepared(cam,packet,source.scene.background,return_radii=True)
                assert p.bits_equal(image,ref['render']) and p.bits_equal(radii,ref['radii'])
                assert image.shape==(3,2208,2064) and torch.isfinite(image).all()
                outputs.append((image,radii))
                checks.append(dict(frame=frame,condition=condition,eye=eye,N=stats['N'],M=stats['M'],K=stats['K'],
                    original_rasterizer_bitwise=True,finite=True,shape='3x2208x2064',
                    ids_sha256='all_map' if ids is None else hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()))
            assert p.bits_equal(outputs[0][0],outputs[1][0]) and p.bits_equal(outputs[0][1],outputs[1][1])
            if eyes[0][2] is not None:assert torch.equal(eyes[0][2],eyes[1][2])
    for _ in range(2):
        for _,frame in views:
            for condition in CONDITIONS:
                for eye in range(2):
                    cam,packet,ids,stats=prepare(frame,condition);raster(cam,packet,condition)
    torch.cuda.synchronize()
    for round_index in range(3):
        ordered=views if round_index%2==0 else list(reversed(views))
        for repeat in range(8):
            for vi,(_,frame) in enumerate(ordered):
                for condition in p.rotate(CONDITIONS,round_index+repeat+vi):
                    for scope in ['prepare_pair_wall','raster_pair_event','raster_pair_wall','pipeline_pair_wall']:
                        if scope.startswith('raster'):eyes=[prepare(frame,condition) for _ in range(2)]
                        torch.cuda.synchronize()
                        if scope=='raster_pair_event':
                            begin=torch.cuda.Event(enable_timing=True);middle=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                            begin.record();a=raster(eyes[0][0],eyes[0][1],condition);middle.record()
                            z=raster(eyes[1][0],eyes[1][1],condition);end.record();end.synchronize()
                            measured=[(scope,begin.elapsed_time(end)),('raster_eye0_event',begin.elapsed_time(middle)),('raster_eye1_event',middle.elapsed_time(end))]
                        else:
                            tick=time.perf_counter_ns()
                            if scope=='prepare_pair_wall':eyes=[prepare(frame,condition) for _ in range(2)]
                            elif scope=='raster_pair_wall':
                                a=raster(eyes[0][0],eyes[0][1],condition);z=raster(eyes[1][0],eyes[1][1],condition)
                            else:
                                eye0=prepare(frame,condition);a=raster(eye0[0],eye0[1],condition)
                                eye1=prepare(frame,condition);z=raster(eye1[0],eye1[1],condition);eyes=[eye0,eye1]
                            torch.cuda.synchronize();measured=[(scope,(time.perf_counter_ns()-tick)/1e6)]
                        st=eyes[0][3];assert st==eyes[1][3]
                        for name,ms in measured:rows.append(dict(condition=condition,frame=frame,round=round_index,repeat=repeat,scope=name,
                            N=st['N'],M=st['M'],K_per_eye=st['K'],K_pair=2*st['K'],ms=ms))
            write('samples.csv',rows);print('PROGRESS',round_index,repeat,len(rows),flush=True)
    assert render.native.stats()[1:3]==[0,0]
    write('checks.csv',checks)
    after={name:h.sha(name) for name in before};changed=[name for name in before if before[name]!=after[name]]
    save('PROTECTED_AFTER.json',after)
    result=dict(passed=not changed,changed=changed,protected_files=len(before),samples=len(rows),checks=len(checks),
        width=2064,height=2208,eyes=2,simulation_only=True,device=torch.cuda.get_device_name(),
        renderer_original_sha_verified=True,actual_K_recorded=True,production_changed=False,training=False,saved_images=0)
    save('VALIDATION.json',result);assert result['passed'];print('COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
