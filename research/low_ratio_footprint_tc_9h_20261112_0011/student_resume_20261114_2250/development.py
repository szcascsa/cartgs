"""仅原开发集Student，先等价和质量检查，再完整当前帧配对测速。"""
from student_study import *
from student_runtime import StudentRenderer
from pose_runtime import PoseRenderer
from pending_runtime import PendingRenderer
from balanced_order import order

CONFIGS = {
    'S_w1': (3, 1, True), 'S_w2': (3, 2, True), 'S_w4': (3, 4, True),
    'P_w2': (4, 2, True), 'H_w2': (5, 2, True), 'P128_w2': (6, 2, True),
    'S_raw': (3, 2, False), 'S_original': (2, 2, True),
}


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
        reference = PoseRenderer(source, student, 38)
        engines = {name: StudentRenderer(source, student, 38, *config) for name, config in CONFIGS.items()}
        engines['G_cpp_pose'] = reference
        engines['F_native'] = PendingRenderer(source, student, 38, native_dispatch=True)
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
            engine = engines[name]
            if name == 'F_native':
                return engine.render(factory.from_source(source, frame), bg, condition)
            return engine.render_from_source(source, frame, bg, condition, factory)

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
                ref_image, ref_radii, ref_ids, _ = results['G_cpp_pose']
                for name, (image, radii, ids, _) in results.items():
                    expected[name, frame] = len(ids)
                    row = dict(scene=spec['id'], condition=condition, frame=frame, variant=name,
                        round=args.round, K=len(ids), same_ids=torch.equal(ids, ref_ids),
                        rgb_bitwise=bits_equal(image, ref_image), radii_bitwise=bits_equal(radii, ref_radii))
                    checks.append(row)
                    if name != 'S_raw':
                        assert row['same_ids'] and row['rgb_bitwise'] and row['radii_bitwise'], row
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
            print('STUDENT_DEVELOPMENT', spec['id'], condition, args.round, flush=True)
        del engines, reference, original, base, source, student
        gc.collect()
        torch.cuda.empty_cache()
    for name, rows in [('samples', samples), ('checks', checks), ('quality', quality)]:
        append(out / (name + '.csv'), rows)
    save(out / 'IDENTITIES.json', identities)
    save(out / 'DONE.json', dict(passed=True, samples=len(samples), checks=len(checks), student_only=True))


if __name__ == '__main__':
    main()
