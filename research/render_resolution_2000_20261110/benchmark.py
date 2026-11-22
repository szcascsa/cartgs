"""仅改变图像宽高的配对渲染测速；冻结CPU学生属性及Gaussian身份。"""
from pathlib import Path
import sys,json,copy,time,hashlib,gc,importlib.util
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PRIOR=Path('/home/prmi/zwj/experiments/cpu_student_new_renderer_20261110')
spec=importlib.util.spec_from_file_location('cpu_resolution_helpers',PRIOR/'benchmark.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
c,torch=b.c,b.torch
SIZES=[(1200,680),(2000,1133)]


def bits(a,z):
    return a.shape==z.shape and a.dtype==z.dtype and torch.equal(a.contiguous().view(torch.uint8),z.contiguous().view(torch.uint8))


@torch.no_grad()
def main():
    reports=HERE/'reports';reports.mkdir(exist_ok=False)
    b.configure();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    protected=json.loads((PRIOR/'reports/PROTECTED_AFTER.json').read_text())
    protected={p:c.sha(p) for p in protected}
    c.save(reports/'PROTECTED_BEFORE.json',protected)
    c.save(reports/'PROTOCOL.json',dict(scenes=b.SLUGS,ratios=b.RATIOS,rounds=2,repeats=12,frames=6,
        sizes=SIZES,pixel_ratio=(2000*1133)/(1200*680),renderer='frozen G capacity Graph',
        compensation='CPU student fast 2/1/1, prepared at original resolution and frozen identically for both',
        control='same camera pose/FoV/projection, same ordered Gaussian IDs, same physical attribute tensors',
        high_resolution='width=2000, height=round(680*2000/1200)=1133; FoV and projection unchanged',
        scopes=['GPU CUDA Event raster','render wall including submission and final synchronization'],
        excluded='CPU inference, upload, model/camera load, graph precompile, GT/metrics/display',
        validation='same-resolution G vs A RGB/radii bitwise; no cross-resolution equality claim',
        scheduling='alternate paired resolution order by frame/repeat/round; reverse condition/frame order in round two',
        training=False,production_changed=False,render_images_saved=False))
    rows=[];checks=[];identities=[];workspace=[]
    for slug in b.SLUGS:
        source=c.load_scene(slug);engine=b.engine_for(source,'student_fast')
        n=len(source['records']);frames=[int(.35*n)+j for j in range(3)]+[int(f*n) for f in [.07,.63,.91]]
        cameras=[c.camera(source['records'][i]) for i in frames]
        assert all((cam.image_width,cam.image_height)==SIZES[0] for cam in cameras)
        gpu={};renderers={};background=torch.zeros(3,device='cuda')
        for size in SIZES:
            gpu[size]=[]
            for camera in cameras:
                cam=copy.copy(camera);cam.image_width,cam.image_height=size
                gpu[size].append(c.cuda_camera(cam))
            renderers[size]=b.create_renderer(gpu[size][0],background,engine.total,use_graph=True)
        full=c.cuda_packet(__import__('renderer').prepare_render_data(source['pc']))
        packets={};metadata={}
        for vi,camera in enumerate(cameras):
            packets['full',vi]=full
            metadata['full',vi]=dict(K=engine.total,M=engine.total,ids_sha256=hashlib.sha256(__import__('numpy').arange(engine.total,dtype='int64').tobytes()).hexdigest())
            for ratio in b.RATIOS:
                packet,ids,info=engine.prepare(camera,ratio_budget=ratio)
                condition=f'local{ratio:g}'
                packets[condition,vi]=c.cuda_packet(packet)
                metadata[condition,vi]=dict(K=len(ids),M=info['candidate_gaussians'],ids_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest())
        conditions=['full']+[f'local{ratio:g}' for ratio in b.RATIOS]
        identities.append(dict(scene=slug,N=engine.total,checkpoint=str(source['checkpoint']),student=str(c.student_path(source)),
            student_sha256=c.sha(c.student_path(source)),teacher_fingerprint=engine.compensator_info['teacher_fingerprint'],frames=frames))
        for condition in conditions:
            for vi,frame in enumerate(frames):
                packet=packets[condition,vi]
                for size in SIZES:
                    cam=gpu[size][vi];image,radii=renderers[size].render(cam,packet,background)
                    ref=b.render_prepared(cam,packet,background,return_radii=True)
                    assert bits(image,ref['render']) and bits(radii,ref['radii']),(slug,condition,frame,size)
                    assert torch.isfinite(image).all()
                    checks.append(dict(scene=slug,condition=condition,frame=frame,width=size[0],height=size[1],
                        K=metadata[condition,vi]['K'],ids_sha256=metadata[condition,vi]['ids_sha256'],rgb_radii_bitwise=True))
        # Warm all tested shapes/plans before collecting any performance sample.
        for _ in range(3):
            for condition in conditions:
                for vi in range(len(frames)):
                    for size in SIZES:renderers[size].render(gpu[size][vi],packets[condition,vi],background)
        torch.cuda.synchronize()
        for round_index in range(2):
            order=conditions if round_index==0 else list(reversed(conditions))
            views=list(range(len(frames))) if round_index==0 else list(reversed(range(len(frames))))
            for repeat in range(12):
                for vi in views:
                    for ci,condition in enumerate(c.rotate(order,repeat+vi)):
                        sizes=SIZES if (repeat+vi+ci+round_index)%2==0 else list(reversed(SIZES))
                        for size in sizes:
                            cam=gpu[size][vi];packet=packets[condition,vi];renderer=renderers[size]
                            gpu_ms,_=b.event_ms(lambda:renderer.render(cam,packet,background))
                            torch.cuda.synchronize();tick=time.perf_counter_ns()
                            renderer.render(cam,packet,background);torch.cuda.synchronize()
                            rows.append(dict(scene=slug,N=engine.total,condition=condition,round=round_index,repeat=repeat,
                                frame=frames[vi],width=size[0],height=size[1],pixels=size[0]*size[1],
                                K=metadata[condition,vi]['K'],gpu_ms=gpu_ms,wall_ms=(time.perf_counter_ns()-tick)/1e6))
                print('PROGRESS',slug,round_index,repeat,len(rows),flush=True)
        for size in SIZES:
            stats=renderers[size].native.stats();assert stats[1:3]==[0,0],stats
            workspace.append(dict(scene=slug,width=size[0],height=size[1],native_stats=stats))
        for name,data in [('samples',rows),('checks',checks)]:c.write_csv(reports/(name+'.csv'),data)
        c.save(reports/'MODEL_IDENTITIES.json',identities);c.save(reports/'RENDERER_WORKSPACE.json',workspace)
        del engine,source,packets,renderers,gpu,packet,renderer,full,image,radii,ref;gc.collect();torch.cuda.empty_cache()
    after={p:c.sha(p) for p in protected};changed=[p for p in protected if protected[p]!=after[p]]
    c.save(reports/'PROTECTED_AFTER.json',after)
    result=dict(passed=not changed,changed=changed,protected_files=len(protected),samples=len(rows),checks=len(checks),
        scenes=len(identities),renderer='frozen G',cpu_inference_timed=False,training=False,production_changed=False,saved_images=0)
    c.save(reports/'VALIDATION.json',result);assert result['passed'];print('COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
