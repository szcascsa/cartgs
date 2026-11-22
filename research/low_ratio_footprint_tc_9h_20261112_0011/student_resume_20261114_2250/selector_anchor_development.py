"""Student GPU实际K成员处理开发集，清零和压缩独立开关。"""
from student_study import *
from joint_final_runtime import FinalStudentRenderer
from selector_anchor_runtime import SelectorAnchorRenderer
from pose_runtime import PoseRenderer
from balanced_order import order


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--round', required=True, type=int)
    args = parser.parse_args()
    out = HERE / 'reports' / args.output
    out.mkdir(exist_ok=False)
    import lpips
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim
    perceptual = lpips.LPIPS(net='alex').cuda().eval()
    samples, checks, quality, identities = [], [], [], []
    specs = student_specs(tune=True)
    if args.round == 2:
        specs.reverse()
    for spec in specs:
        source, student, path = load_case(spec)
        identities.append(identity(spec, source, student, path))
        bg = source.scene.background
        factory = NativeCameraPreparer(p.MiniCam)
        engines={'I_final':FinalStudentRenderer(source,student,182,True,True,True,15),'anchor':SelectorAnchorRenderer(source,student,182,True,True,True,15,True),'control':SelectorAnchorRenderer(source,student,182,True,True,True,15,False)}
        original = network_variants(source, student)['fp32'][0]
        base = BaselineRenderer()
        views = motions(source, quick=True)
        if args.round == 2:
            views.reverse()
        cam = factory.from_source(source, views[0][1])
        for engine in engines.values():
            engine.precompile(cam, bg)
        base.precompile(source.runtime.total, cam, bg)

        def render(name, frame, condition):
            return engines[name].render_from_source(source, frame, bg, condition, factory)

        conditions = ['local0.01', 'local0.05', 'local0.1', 'local0.2', 'local0.3',
                      'local0.4', 'local0.5', 'local0.7', 'fixed0', 'fixed1', 'full']
        if args.round == 2:
            conditions.reverse()
        for ci, condition in enumerate(conditions):
            expected = {}
            for motion, frame in views:
                cam = factory.from_source(source, frame)
                gt = source.scene.ground_truth(frame).clamp(0, 1)
                packet, _, _ = p.request(original, cam, condition)
                base_image, _ = base.render(cam, packet, bg)
                baseline_psnr = float(-10 * torch.log10((base_image.clamp(0, 1) - gt).square().mean().clamp_min(1e-30)))
                results = {name: render(name, frame, condition) for name in engines}
                ref_image, ref_radii, ref_ids, _ = results['I_final']
                for name, (image, radii, ids, _) in results.items():
                    expected[name, frame] = len(ids)
                    row = dict(scene=spec['id'], condition=condition, frame=frame, variant=name,
                        round=args.round, K=len(ids), same_ids=torch.equal(ids, ref_ids),
                        rgb_bitwise=bits_equal(image, ref_image), radii_bitwise=bits_equal(radii, ref_radii),
                        rgb_max_error=float((image-ref_image).abs().max()),
                        radii_max_error=int((radii-ref_radii).abs().max()) if len(radii) else 0)
                    checks.append(row)
                    assert row['same_ids'] and bool(torch.isfinite(image).all()), row
                    assert row['rgb_bitwise'] and row['radii_bitwise'], row
                    x = image.clamp(0, 1)
                    psnr = float(-10 * torch.log10((x - gt).square().mean().clamp_min(1e-30)))
                    quality.append(dict(scene=spec['id'], condition=condition, frame=frame, variant=name,
                        round=args.round, psnr_loss=baseline_psnr - psnr,
                        ssim=float(ssim(x[None], gt[None], data_range=1.)),
                        lpips=float(perceptual(x[None] * 2 - 1, gt[None] * 2 - 1))))
                del results, gt
            for repeat in range(12):
                for j, (motion, frame) in enumerate(views):
                    for name in order(list(engines), repeat + j + ci, reverse=args.round == 2):
                        torch.cuda.synchronize()
                        start = time.perf_counter_ns()
                        _, _, ids, _ = render(name, frame, condition)
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter_ns() - start) * 1e-6
                        assert len(ids) == expected[name, frame]
                        samples.append(dict(scene=spec['id'], condition=condition, frame=frame,
                            motion=motion, variant=name, round=args.round, repeat=repeat,
                            N=source.runtime.total, K=len(ids), ms=elapsed))
            print('MEMBERS_DEVELOPMENT', spec['id'], condition, args.round, flush=True)
        del engines, original, base, source, student
        gc.collect()
        torch.cuda.empty_cache()
    for name, rows in [('samples', samples), ('checks', checks), ('quality', quality)]:
        append(out / (name + '.csv'), rows)
    save(out / 'IDENTITIES.json', identities)
    save(out / 'DONE.json', dict(passed=True, samples=len(samples), checks=len(checks), student_only=True))


if __name__ == '__main__':
    main()
