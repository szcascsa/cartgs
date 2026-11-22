"""冻结CPU推理配新G光栅器；CPU、上传、GPU与完整延迟分别实测。"""
from pathlib import Path
import sys, json, hashlib, importlib.util, time, copy, gc, argparse
sys.dont_write_bytecode = True
ROOT = Path('/home/prmi/zwj/experiments')
CPU = ROOT/'cpu_student_fast_20261110_1519'
G = ROOT/'sparse_rasterizer_sixhour_20261110_0053'
sys.path.insert(0, str(CPU))
import common as c
import kernel_runtime, mid_runtime
import torch
import numpy as np
from PIL import Image
sys.path.insert(0, str(G))
from frozen_renderer import create_renderer
from renderer import render_prepared
HERE = Path(__file__).resolve().parent
RATIOS = [.01, .05, .1, .2, .3, .4, .5, .7]
SLUGS = ['Replica__office1__mono', 'Replica__office2__rgbd', 'Replica__room0__rgbd', 'Replica__room2__mono']


def verified_module(name, path, sha):
    assert c.sha(path) == sha, str(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def configure():
    frozen = json.loads((CPU/'reports/FROZEN_CANDIDATE.json').read_text())
    for name, expected in frozen['sources'].items():
        assert c.sha(CPU/name) == expected, name
    fast = verified_module('mycartgs_cpu_student_fast', Path(frozen['binary']), frozen['binary_sha256'])
    baseline = verified_module('mycartgs_cpu_mid', Path(frozen['baseline_binary']), frozen['baseline_binary_sha256'])
    kernel_runtime.extension = lambda: fast
    mid_runtime.extension = lambda: baseline
    return fast, baseline


def engine_for(source, variant):
    if variant == 'teacher_cpu':
        return c.CpuRenderer(source['model'], source['pc'], source['reference'].noise, 6)
    return c.native(source, 'baseline' if variant == 'student_cpu' else 'context')


def camera_buffer(cam):
    return torch.cat([getattr(cam, name).flatten() for name in
        ['world_view_transform', 'projection_matrix', 'full_proj_transform', 'camera_center']])


def uploaded_camera(cam, device_buffer):
    result = copy.copy(cam)
    for i, name in enumerate(['world_view_transform', 'projection_matrix', 'full_proj_transform']):
        setattr(result, name, device_buffer[16*i:16*(i+1)].view(4,4))
    result.camera_center = device_buffer[48:51]
    return result


def event_ms(function):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record(); result = function(); end.record(); end.synchronize()
    return start.elapsed_time(end), result


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(); p.add_argument('--smoke', action='store_true'); args = p.parse_args()
    reports = HERE/('smoke' if args.smoke else 'reports'); reports.mkdir(exist_ok=False)
    fast, baseline = configure()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    protect = json.loads((CPU/'reports/PROTECTED_BEFORE.json').read_text())
    protect.update(json.loads((ROOT/'sparse_student_lod_benefit_20261110/reports/PROTECTED_AFTER.json').read_text()))
    protect = {path:c.sha(path) for path in protect}
    c.save(reports/'PROTECTED_BEFORE.json', protect)
    c.save(reports/'PROTOCOL.json', dict(ratios=RATIOS, scenes=SLUGS, rounds=2, cpu_threads=6,
        model='same teacher/map/student fingerprints as CPU specialized report', renderer='frozen G',
        full='GPU-resident frozen Master, upload 51 camera floats per view',
        cpu='untimed camera construction; native prepare includes candidates, Selector, Compensator, decode/pack',
        stages=['frustum_candidates_context', 'optional_context_noop', 'selector_score', 'selection', 'compensation_decode_pack'],
        scopes=['cpu_prepare', 'cpu_stages', 'h2d_event', 'raster_event', 'raster_wall', 'pipeline_wall'],
        upload='isolated single contiguous pinned camera+attributes H2D, not subtraction of pipeline times',
        pipeline='native three-slot H2D with dependency + G render + final sync; no frame overlap',
        timing='CPU and GPU clocks separated; separately sampled medians are not additive',
        camera_samples='3 continuous and 3 jumps from saved training cameras, not entire trajectory',
        training=False, production_changed=False, images_saved=False, smoke=args.smoke))
    rows=[]; checks=[]; qualities=[]; identities=[]
    slugs=SLUGS[:1] if args.smoke else SLUGS
    ratios=[.01,.3] if args.smoke else RATIOS
    for si, slug in enumerate(slugs):
        source=c.load_scene(slug); n=len(source['records'])
        frames=[int(.35*n)+j for j in range(3)]+[int(f*n) for f in [.07,.63,.91]]
        frames=frames[:2] if args.smoke else frames
        cams=[c.camera(source['records'][f]) for f in frames]
        gpu_cams=[c.cuda_camera(cam) for cam in cams]
        full_gpu=c.cuda_packet(__import__('renderer').prepare_render_data(source['pc']))
        background=torch.zeros(3,device='cuda')
        renderer=create_renderer(gpu_cams[0],background,len(source['pc'].get_xyz),use_graph=True)
        ids_expected={}; student_attributes={}; scores_expected={}
        gts=[]
        for f in frames:
            path=Path(source['spec']['dataset'])/'results'/Path(source['records'][f]['img_name']).name
            gt=torch.from_numpy(np.asarray(Image.open(path).convert('RGB')).copy()).permute(2,0,1).cuda().float()/255
            gts.append(gt)
        identities.append(dict(scene=slug,N=len(source['pc'].get_xyz),checkpoint=str(source['checkpoint']),
            teacher_sha256=c.sha(source['checkpoint']/'selector.pt'),ply_sha256=c.sha(source['ply']),
            student=str(c.student_path(source)),student_sha256=c.sha(c.student_path(source)),frames=frames))
        for round_index in range(2):
            variants=c.rotate(['teacher_cpu','student_cpu','student_fast'],si+round_index)
            for variant in variants:
                engine=engine_for(source,variant)
                assert engine.compensator_info['parameters']==(25742 if variant=='teacher_cpu' else 3534)
                module=baseline if variant=='student_cpu' else fast
                handoff=module.NativeHandoff(engine.native,engine.sh_degree)
                cpu_flat=torch.empty(engine.total*engine.width+51,pin_memory=True)
                device_flat=torch.empty_like(cpu_flat,device='cuda')
                for ratio in ratios:
                    # Exact input identity, finite outputs and same-packet A/G raster parity.
                    for vi, cam in enumerate(cams):
                        packet,ids,info=engine.prepare(cam,ratio_budget=ratio)
                        key=(ratio,vi); id_sha=hashlib.sha256(ids.numpy().tobytes()).hexdigest()
                        if key in ids_expected: assert id_sha==ids_expected[key]
                        else: ids_expected[key]=id_sha
                        score=engine.native.diagnostics()[2]
                        score_sha=hashlib.sha256(score.numpy().tobytes()).hexdigest() if len(ids) else 'unused'
                        if key in scores_expected: assert score_sha==scores_expected[key]
                        else: scores_expected[key]=score_sha
                        for name,value in packet['inputs'].items():
                            if isinstance(value,torch.Tensor): assert torch.isfinite(value).all()
                        attribute_errors={}
                        if variant!='teacher_cpu':
                            if key not in student_attributes:
                                student_attributes[key]={k:v.clone() for k,v in packet['inputs'].items() if isinstance(v,torch.Tensor)}
                            for name, expected in student_attributes[key].items():
                                value=packet['inputs'][name]
                                attribute_errors['max_'+name]=float((value-expected).abs().max()) if value.numel() else 0.
                                if name in ['means3D','dc','shs','means2D']: assert torch.equal(value,expected),name
                        gpu_packet=c.cuda_packet(packet)
                        image,radii=renderer.render(gpu_cams[vi],gpu_packet,background)
                        ref=render_prepared(gpu_cams[vi],gpu_packet,background,return_radii=True)
                        assert torch.equal(image.view(torch.uint8),ref['render'].view(torch.uint8))
                        assert torch.equal(radii,ref['radii'])
                        qualities.append(dict(scene=slug,round=round_index,variant=variant,ratio=ratio,frame=frames[vi],
                            K=len(ids),psnr=float(-10*torch.log10((image.clamp(0,1)-gts[vi]).square().mean().clamp_min(1e-12)))))
                        ticket=handoff.prepare(cam,ratio_budget=ratio); ticket_packet=ticket.wait()
                        hand_image,hand_radii=renderer.render(ticket.camera,ticket_packet,background)
                        assert torch.equal(image.view(torch.uint8),hand_image.view(torch.uint8)) and torch.equal(radii,hand_radii)
                        assert torch.equal(ids,ticket.ids); ticket.finish(); torch.cuda.synchronize()
                        checks.append(dict(scene=slug,round=round_index,variant=variant,ratio=ratio,frame=frames[vi],
                            K=len(ids),M=info['candidate_gaussians'],ids_sha256=id_sha,score_sha256=score_sha,raster_bitwise=True,handoff_bitwise=True,
                            attribute_errors=json.dumps(attribute_errors)))
                    count=6 if args.smoke else 48
                    # CPU only: no GPU calls or synchronization inside the timed interval.
                    for j in range(6): engine.prepare(cams[j%len(cams)],ratio_budget=ratio)
                    for j in range(count):
                        vi=j%len(cams); tick=time.perf_counter_ns()
                        packet,ids,info=engine.prepare(cams[vi],ratio_budget=ratio)
                        rows.append(dict(scene=slug,round=round_index,variant=variant,ratio=ratio,frame=frames[vi],
                            N=engine.total,M=info['candidate_gaussians'],K=len(ids),scope='cpu_prepare',sample=j,ms=(time.perf_counter_ns()-tick)/1e6))
                    for j in range(len(cams)*2):
                        vi=j%len(cams); _,ids,info=engine.prepare(cams[vi],ratio_budget=ratio,timing=True)
                        for stage,value in enumerate(info['stage_ms']):
                            rows.append(dict(scene=slug,round=round_index,variant=variant,ratio=ratio,frame=frames[vi],N=engine.total,
                                M=info['candidate_gaussians'],K=len(ids),scope='stage_'+str(stage),sample=j,ms=value))
                    # GPU-only diagnostics: CPU prepare and camera packing precede Event interval.
                    for vi, cam in enumerate(cams):
                        flat,ids,info=engine.prepare(cam,ratio_budget=ratio,flat_output=True,output=cpu_flat[:-51])
                        size=flat.numel(); cpu_flat[size:size+51].copy_(camera_buffer(cam)); host=cpu_flat[:size+51];device=device_flat[:size+51]
                        gpu_packet=engine.packet(device[:size],len(ids)); gpu_cam=uploaded_camera(cam,device[size:])
                        device.copy_(host,non_blocking=True);torch.cuda.synchronize()
                        for j in range(2 if args.smoke else 6):
                            ms,_=event_ms(lambda:device.copy_(host,non_blocking=True))
                            rows.append(dict(scene=slug,round=round_index,variant=variant,ratio=ratio,frame=frames[vi],N=engine.total,M=info['candidate_gaussians'],K=len(ids),scope='h2d_event',sample=j,ms=ms))
                            ms,_=event_ms(lambda:renderer.render(gpu_cam,gpu_packet,background))
                            rows.append(dict(scene=slug,round=round_index,variant=variant,ratio=ratio,frame=frames[vi],N=engine.total,M=info['candidate_gaussians'],K=len(ids),scope='raster_event',sample=j,ms=ms))
                            torch.cuda.synchronize();tick=time.perf_counter_ns();renderer.render(gpu_cam,gpu_packet,background);torch.cuda.synchronize()
                            rows.append(dict(scene=slug,round=round_index,variant=variant,ratio=ratio,frame=frames[vi],N=engine.total,M=info['candidate_gaussians'],K=len(ids),scope='raster_wall',sample=j,ms=(time.perf_counter_ns()-tick)/1e6))
                    for j in range(6):
                        ticket=handoff.prepare(cams[j%len(cams)],ratio_budget=ratio);renderer.render(ticket.camera,ticket.wait(),background);ticket.finish()
                    torch.cuda.synchronize()
                    for j in range(count):
                        vi=j%len(cams);torch.cuda.synchronize();tick=time.perf_counter_ns()
                        ticket=handoff.prepare(cams[vi],ratio_budget=ratio)
                        renderer.render(ticket.camera,ticket.wait(),background);ticket.finish();torch.cuda.synchronize()
                        rows.append(dict(scene=slug,round=round_index,variant=variant,ratio=ratio,frame=frames[vi],N=engine.total,M=ticket.stats['candidate_gaussians'],K=len(ticket.ids),scope='pipeline_wall',sample=j,ms=(time.perf_counter_ns()-tick)/1e6))
                    print('PROGRESS',slug,round_index,variant,ratio,len(rows),flush=True)
                handoff.synchronize();del handoff,engine,ticket,packet,gpu_packet,flat,ids,device,host;gc.collect()
            # Full map is resident: no network inference and no per-frame map upload.
            host_camera=torch.empty(51,pin_memory=True);device_camera=torch.empty(51,device='cuda')
            for vi,cam in enumerate(cams):
                host_camera.copy_(camera_buffer(cam));device_camera.copy_(host_camera,non_blocking=True)
                gpu_cam=uploaded_camera(cam,device_camera);torch.cuda.synchronize()
                image,radii=renderer.render(gpu_cam,full_gpu,background);ref=render_prepared(gpu_cam,full_gpu,background,return_radii=True)
                assert torch.equal(image.view(torch.uint8),ref['render'].view(torch.uint8)) and torch.equal(radii,ref['radii'])
                qualities.append(dict(scene=slug,round=round_index,variant='full',ratio=1.,frame=frames[vi],K=len(radii),psnr=float(-10*torch.log10((image.clamp(0,1)-gts[vi]).square().mean().clamp_min(1e-12)))))
                for j in range(6 if args.smoke else 24):
                    for scope in ['h2d_event','raster_event','raster_wall','pipeline_wall']:
                        if scope=='h2d_event':ms,_=event_ms(lambda:device_camera.copy_(host_camera,non_blocking=True))
                        elif scope=='raster_event':ms,_=event_ms(lambda:renderer.render(gpu_cam,full_gpu,background))
                        else:
                            torch.cuda.synchronize();tick=time.perf_counter_ns()
                            if scope=='pipeline_wall':device_camera.copy_(host_camera,non_blocking=True)
                            renderer.render(gpu_cam,full_gpu,background);torch.cuda.synchronize();ms=(time.perf_counter_ns()-tick)/1e6
                        rows.append(dict(scene=slug,round=round_index,variant='full',ratio=1.,frame=frames[vi],N=len(radii),M=len(radii),K=len(radii),scope=scope,sample=j,ms=ms))
            c.write_csv(reports/'samples.csv',rows);c.write_csv(reports/'checks.csv',checks);c.write_csv(reports/'quality.csv',qualities)
        assert renderer.native.stats()[1:3]==[0,0]
        c.save(reports/'MODEL_IDENTITIES.json',identities)
        del source,renderer,full_gpu,gts,student_attributes;gc.collect();torch.cuda.empty_cache()
    after={path:c.sha(path) for path in protect};changed=[path for path in protect if protect[path]!=after[path]]
    c.save(reports/'PROTECTED_AFTER.json',after)
    result=dict(passed=not changed,changed=changed,protected_files=len(protect),samples=len(rows),checks=len(checks),
        quality_rows=len(qualities),scenes=len(slugs),saved_images=0,training=False,production_changed=False)
    c.save(reports/'VALIDATION.json',result);assert result['passed'];print('COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
