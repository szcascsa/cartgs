"""冻结Student最终四组及独立关闭归因；当前相机每次进入，质量、诊断与计时分离。"""
from joint_final_study import *
from joint_release_runtime import JointReleaseRenderer
from tail_strong_runtime import TailStrongRenderer
from tail_cutoff_runtime import TailCutoffRenderer
from balanced_order import order


def release_engines(source, student, mode):
    networks = network_variants(source, student)
    factory = networks['tensor_best'][1]
    raster = BaselineRenderer()
    new_raster = (TailStrongRenderer(mode) if mode in [301, 302] else
                  TailCutoffRenderer(203) if mode == 203 else EncodedSignRenderer(182))
    config = dict(reuse=True, fused=True, finish=True)
    dispatch = dict(C_native_F2=PoseRenderer(source, student, 0),
        I_previous=FinalStudentRenderer(source, student, 182, **config, members=15),
        new_prepare_182=JointReleaseRenderer(source, student, 182, **config),
        D_final=JointReleaseRenderer(source, student, mode, **config),
        no_direct=JointReleaseRenderer(source, student, mode, **config, direct=False),
        no_grouped=JointReleaseRenderer(source, student, mode, **config, grouped=False),
        no_native_full=JointReleaseRenderer(source, student, mode, **config, native_full=False))
    return networks, factory, raster, new_raster, dispatch


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--group', type=int, required=True)
    parser.add_argument('--round', type=int, default=1)
    parser.add_argument('--quality', action='store_true')
    args = parser.parse_args()
    out = HERE / 'reports' / args.output
    out.mkdir(exist_ok=False)
    mode = json.loads((HERE / 'reports/SELECTED_RASTER_20261115.json').read_text())['mode']
    rows, events, checks, quality, identities = [], [], [], [], []
    if args.quality:
        import lpips
        from torchmetrics.functional.image import structural_similarity_index_measure as ssim
        perceptual = lpips.LPIPS(net='alex').cuda().eval()

        def metrics(image, gt):
            image = image.clamp(0, 1)
            return dict(psnr=float(-10 * torch.log10((image - gt).square().mean().clamp_min(1e-30))),
                ssim=float(ssim(image[None], gt[None], data_range=1.)),
                lpips=float(perceptual(image[None] * 2 - 1, gt[None] * 2 - 1)))
    specs = student_specs()[args.group * 2:args.group * 2 + 2]
    if args.round == 2:
        specs.reverse()
    for spec in specs:
        source, student, path = load_case(spec)
        identities.append(identity(spec, source, student, path))
        networks, factory, raster, new_raster, dispatch = release_engines(source, student, mode)
        bg, views = source.scene.background, motions(source)
        if args.round == 2:
            views.reverse()
        cam = factory.from_source(source, views[0][1])
        for engine in [raster, new_raster]:
            engine.precompile(source.runtime.total, cam, bg)
        for engine in dispatch.values():
            engine.precompile(cam, bg)
        names = ['A_fp32_F2', 'B_fp32_new', *dispatch]

        def render(name, frame, condition):
            if name in dispatch:
                return dispatch[name].render_from_source(source, frame, bg, condition, factory)
            camera = factory.from_source(source, frame)
            packet, ids, stats = p.request(networks['fp32'][0], camera, condition)
            image, radii = (raster if name == 'A_fp32_F2' else new_raster).render(camera, packet, bg)
            return image, radii, ids, stats

        for ci, condition in enumerate(CONDITIONS if args.round == 1 else CONDITIONS[::-1]):
            expected = {}
            for motion, frame in views:
                values = {name: render(name, frame, condition) for name in names}
                final = values['D_final']
                for name, (image, radii, ids, stats) in values.items():
                    assert len(ids) == len(final[2]) and bool(torch.isfinite(image).all())
                    expected[name, frame] = len(ids)
                    if name in ['no_direct', 'no_grouped', 'no_native_full']:
                        check = dict(scene=spec['id'], condition=condition, frame=frame, variant=name,
                            round=args.round, K=len(ids), same_ids=torch.equal(ids, final[2]),
                            rgb_bitwise=bits_equal(image, final[0]), radii_bitwise=bits_equal(radii, final[1]))
                        checks.append(check)
                        assert check['same_ids'] and check['rgb_bitwise'] and check['radii_bitwise'], check
                cam = factory.from_source(source, frame)
                packet, ids, _ = p.request(networks['tensor_best'][0], cam, condition)
                split = new_raster.render(cam, packet, bg)
                assert torch.equal(ids, final[2]) and bits_equal(split[0], final[0]) and bits_equal(split[1], final[1]), (spec['id'], condition, 'pure vs joined')
                if args.quality:
                    gt = source.scene.ground_truth(frame).clamp(0, 1)
                    baseline = metrics(values['A_fp32_F2'][0], gt)
                    for name, (image, radii, ids, _) in values.items():
                        m = metrics(image, gt)
                        quality.append(dict(scene=spec['id'], condition=condition, frame=frame, variant=name,
                            K=len(ids), psnr_loss=baseline['psnr'] - m['psnr'], baseline_psnr=baseline['psnr'],
                            baseline_ssim=baseline['ssim'], baseline_lpips=baseline['lpips'], **m,
                            id_overlap=float(torch.isin(ids, values['A_fp32_F2'][2]).float().mean()) if len(ids) else 1.))
            if not args.quality:
                for repeat in range(10):
                    for index, (motion, frame) in enumerate(views):
                        for name in order(names, repeat + index + ci, reverse=args.round == 2):
                            torch.cuda.synchronize()
                            start = time.perf_counter_ns()
                            result = render(name, frame, condition)
                            torch.cuda.synchronize()
                            rows.append(dict(scene=spec['id'], condition=condition, frame=frame, motion=motion,
                                variant=name, round=args.round, repeat=repeat, K=expected[name, frame],
                                N=source.runtime.total, ms=(time.perf_counter_ns() - start) * 1e-6))
                for _, frame in [views[0], views[-1]]:
                    for repeat in range(6):
                        for name in order(names, repeat + ci, reverse=args.round == 2):
                            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            start.record()
                            render(name, frame, condition)
                            end.record()
                            end.synchronize()
                            events.append(dict(scene=spec['id'], condition=condition, frame=frame, variant=name,
                                round=args.round, repeat=repeat, cuda_ms=start.elapsed_time(end)))
            print('RELEASE_JOINT', spec['id'], condition, args.round, args.quality, flush=True)
        del networks, dispatch, raster, new_raster, source, student, values
        gc.collect()
        torch.cuda.empty_cache()
    for name, data in [('samples', rows), ('events', events), ('checks', checks), ('quality', quality)]:
        append(out / (name + '.csv'), data)
    save(out / 'IDENTITIES.json', identities)
    save(out / 'DONE.json', dict(passed=True, samples=len(rows), checks=len(checks), quality=len(quality), no_images=True))


if __name__ == '__main__':
    main()
