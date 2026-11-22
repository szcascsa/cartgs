"""跨场景同身份配对质量评价，不保存渲染图片。"""
from scene_runtime import *
import csv
import math
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from compensation_student import load_student, decode_student, distillation_loss
from deployment_runtime import DeploymentRenderer
from distill_compensation import render_student, student_residual


@torch.no_grad()
def main():
    slug = sys.argv[1]
    scene, source = load_scene(slug)
    folder = ROOT/'reports'/slug
    preflight = json.loads((folder/'PREFLIGHT.json').read_text())
    assert preflight['fingerprint'] == source.fingerprint
    paths = {stage:ROOT/'models'/slug/stage/'best.pt'
        for stage in ['image_calibration', 'geometry_calibration', 'continuous_calibration']}
    students, runtimes, identities = {}, {}, {}
    for name,path in paths.items():
        students[name], payload = load_student(path, 'cuda', source.fingerprint)
        runtimes[name] = DeploymentRenderer(source.scene.model,source.scene.gaussians,source.scene.config,student=students[name])
        identities[name] = dict(path=str(path),sha256=sha(path),step=payload['step'],
            validation=payload['metadata']['best_validation'], configuration=payload['metadata']['configuration'])
    save(folder/'EVALUATION_CHECKPOINTS.json',identities)
    perceptual = LearnedPerceptualImagePatchSimilarity(normalize=True,reduction='none').cuda()
    metrics_rows = []
    attribute_rows = []
    full_rows = []
    full_packet,_ = source.runtime.full()
    ratios = [.01,.05,.1,.3,.5,.7,.03,.08,.2,.4,.6]
    csv_path = folder/'QUALITY_PER_VIEW.csv'
    assert not csv_path.exists()
    writer = None
    with csv_path.open('x',newline='') as stream:
        for frame in preflight['test_views']:
            gt = source.scene.ground_truth(frame).clamp(0,1)
            cam = source.camera(frame)
            full_image = render_prepared(cam,full_packet,source.scene.background)['render'].clamp(0,1)
            full_psnr = float(-10*torch.log10((full_image-gt).square().mean().clamp_min(1e-12)))
            full_rows.append(dict(frame=frame,gaussians=scene['gaussians'],psnr=full_psnr))
            for ratio in ratios:
                for local in source.local_modes:
                    case = source.sample(frame,ratio,local)
                    teacher_image = render_prepared(cam,case['packet'],source.scene.background)['render']
                    images = [teacher_image]
                    teacher_ids_hash = sha_ids(case['ids'])
                    teacher_attributes = decode_student(case['target'],case['master'],case['softplus'])
                    for name,student in students.items():
                        residual = student_residual(student,case)
                        if local:
                            packet, ids, stats = runtimes[name].prepare(cam,ratio_budget=ratio)
                            assert torch.equal(ids,case['ids']),(slug,frame,ratio,name)
                            assert len(ids)==min(stats['requested_budget'],stats['eligible_candidates'])
                            image = render_prepared(cam,packet,source.scene.background)['render']
                        else:
                            image = render_student(student,case,source.scene.background,
                                source.runtime._base.pc.active_sh_degree,residual)
                        assert torch.isfinite(image).all()
                        images.append(image)
                        attributes = decode_student(residual,case['master'],case['softplus'])
                        if any(not torch.isfinite(v).all() for v in attributes.values() if isinstance(v,torch.Tensor)):
                            raise RuntimeError('Nonfinite student attribute in evaluation')
                        losses = distillation_loss(student,case,residual)
                        shift = residual[:,:3].norm(dim=1)/case['reference'].flatten()
                        error = (residual[:,:3]-case['target'][:,:3]).norm(dim=1)/case['reference'].flatten()
                        scale_ratio = (attributes['scales']/teacher_attributes['scales']).flatten()
                        attribute_rows.append(dict(student=name,frame=frame,ratio=ratio,local=local,K=len(case['ids']),
                            xyz_p99=float(torch.quantile(shift,.99)),xyz_max=float(shift.max()),
                            xyz_error_p99=float(torch.quantile(error,.99)),
                            scale_ratio_p01=float(torch.quantile(scale_ratio,.01)),
                            scale_ratio_p99=float(torch.quantile(scale_ratio,.99)),scale_ratio_max=float(scale_ratio.max()),
                            **{key:float(value) for key,value in losses.items()}))
                    names = ['teacher', *students]
                    raw_images = torch.stack(images)
                    predictions = raw_images.clamp(0,1)
                    targets = gt.unsqueeze(0).expand_as(predictions)
                    psnrs = -10*torch.log10((predictions-targets).square().mean((1,2,3)).clamp_min(1e-12))
                    structure = structural_similarity_index_measure(predictions,targets,data_range=1.,reduction='none')
                    distance = perceptual(predictions,targets).flatten()
                    perceptual.reset()
                    values = torch.stack([psnrs,structure,distance],dim=1)
                    assert torch.isfinite(values).all()
                    for index,(name,value) in enumerate(zip(names,values.cpu().tolist())):
                        row = dict(scene=scene['id'],student=name,frame=frame,ratio=ratio,
                            ratio_kind='anchor' if ratio in [.01,.05,.1,.3,.5,.7] else 'non_anchor',
                            local=local,N=scene['gaussians'],K=len(case['ids']),selection_sha256=teacher_ids_hash,
                            candidate_N=stats['candidate_gaussians'] if local else scene['gaussians'],
                            eligible_N=stats['eligible_candidates'] if local else None,
                            nominal_target_K=stats['requested_budget'] if local else math.floor(ratio*scene['gaussians']),
                            psnr=value[0],ssim=value[1],lpips=value[2],
                            psnr_loss=float(psnrs[0])-value[0],
                            teacher_student_mse=float((raw_images[index]-teacher_image).square().mean()))
                        metrics_rows.append(row)
                        if writer is None:
                            writer = csv.DictWriter(stream,fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
            stream.flush()
            print('CROSS_EVAL',scene['id'],frame,'rows',len(metrics_rows),flush=True)
    for name,rows in [('ATTRIBUTES_PER_VIEW.csv',attribute_rows),('FULL_PER_VIEW.csv',full_rows)]:
        with (folder/name).open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summaries = []
    for name in ['teacher',*students]:
        for ratio in ratios:
            for local in source.local_modes:
                group = [r for r in metrics_rows if r['student']==name and r['ratio']==ratio and r['local']==local]
                losses = [r['psnr_loss'] for r in group]
                summaries.append(dict(scene=scene['id'],student=name,ratio=ratio,local=local,views=len(group),
                    psnr=float(np.mean([r['psnr'] for r in group])),ssim=float(np.mean([r['ssim'] for r in group])),
                    lpips=float(np.mean([r['lpips'] for r in group])),mean_loss=float(np.mean(losses)),
                    p99_loss=float(np.percentile(losses,99)),worst_loss=max(losses),
                    teacher_student_mse=float(np.mean([r['teacher_student_mse'] for r in group])),
                    K_mean=float(np.mean([r['K'] for r in group])),K_min=min(r['K'] for r in group),K_max=max(r['K'] for r in group),
                    gate=bool(np.mean(losses)<=.01 and np.percentile(losses,99)<=.05 and max(losses)<=.1)))
    with (folder/'QUALITY_SUMMARY.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    save(folder/'QUALITY_SUMMARY.json',dict(scene=scene,test_views=preflight['test_views'],
        validation_views=preflight['validation_views'],identities=identities,summaries=summaries,
        anchor_count=len(source.runtime._base.anchors.centers),
        selector_ids_identical=True,full_psnr=float(np.mean([r['psnr'] for r in full_rows])),
        frames=len(preflight['test_views']),images_saved=0,speed_measured=False))
    print('CROSS_EVALUATION_COMPLETE',scene['id'],len(metrics_rows),flush=True)


def sha_ids(ids):
    return hashlib.sha256(ids.contiguous().cpu().numpy().tobytes()).hexdigest()


if __name__ == '__main__':
    main()
