"""匹配Student全场景：A/B/C/D与独立关闭归因；质量和正式计时分离。"""
from student_final_study import *
from balanced_order import order


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--group', required=True, type=int)
    parser.add_argument('--round', type=int, default=1)
    parser.add_argument('--quality', action='store_true')
    args = parser.parse_args()
    out = HERE/'reports'/args.output
    out.mkdir(exist_ok=False)
    specs = student_specs()[args.group*2:args.group*2+2]
    if args.round == 2:
        specs.reverse()
    samples, events, checks, quality, memory, identities = [], [], [], [], [], []
    if args.quality:
        import lpips
        from torchmetrics.functional.image import structural_similarity_index_measure as ssim
        perceptual = lpips.LPIPS(net='alex').cuda().eval()

        def metrics(image, gt):
            assert bool(torch.isfinite(image).all())
            image = image.clamp(0, 1)
            return dict(psnr=float(-10*torch.log10((image-gt).square().mean().clamp_min(1e-30))),
                ssim=float(ssim(image[None], gt[None], data_range=1.)),
                lpips=float(perceptual(image[None]*2-1, gt[None]*2-1)))

    for spec in specs:
        source, student, path = load_case(spec)
        identities.append(identity(spec, source, student, path))
        tick = time.perf_counter_ns()
        torch.cuda.reset_peak_memory_stats()
        networks, renders, variants, dispatch = final_engines(source, student)
        factory = networks['tensor_best'][1]
        bg = source.scene.background
        views = motions(source)
        if args.round == 2:
            views.reverse()
        cam = factory.from_source(source, views[0][1])
        for engine in renders.values():
            engine.precompile(source.runtime.total, cam, bg)
        for engine in dispatch.values():
            engine.precompile(cam, bg)
        names = list(variants) + list(dispatch)
        memory.append(dict(scene=spec['id'], round=args.round, startup_ms=(time.perf_counter_ns()-tick)*1e-6,
            all_variants_peak_bytes=torch.cuda.max_memory_allocated(), all_variants_live_bytes=torch.cuda.memory_allocated()))
        for name, engine in dispatch.items():
            memory.append(dict(scene=spec['id'], variant=name, round=args.round,
                raster_workspace_bytes=engine.native.workspace_bytes(),
                projection_bytes=engine.network._pipeline._preparation_bytes(),
                tensor_bytes=engine.network._pipeline._tensor_bytes()))

        def render(name, frame, condition):
            if name in dispatch:
                return dispatch_frame(name, dispatch[name], source, frame, bg, condition, factory)
            (network, camera), renderer = variants[name]
            camera = camera.from_source(source, frame)
            packet, ids, stats = p.request(network, camera, condition)
            image, radii = renderer.render(camera, packet, bg)
            return image, radii, ids, stats

        conditions = CONDITIONS if args.round == 1 else CONDITIONS[::-1]
        for ci, condition in enumerate(conditions):
            expected = {}
            for motion, frame in views:
                results = {name: render(name, frame, condition) for name in names}
                reference = results['D_tensor_best_new']
                baseline_ids = results['A_fp32_F2'][2]
                for name, (image, radii, ids, stats) in results.items():
                    expected[name, frame] = len(ids)
                    assert len(ids) == len(baseline_ids), (spec['id'], condition, name, 'strict K')
                    if name in dispatch:
                        pair = results['C_tensor_best_F2'] if name == 'H_alternative37' else reference
                        if name == 'H_alternative37':
                            assert torch.equal(ids, pair[2])
                        else:
                            row = dict(scene=spec['id'], condition=condition, frame=frame, round=args.round,
                                variant=name, K=len(ids), same_ids=torch.equal(ids, pair[2]),
                                rgb_bitwise=bits_equal(image, pair[0]), radii_bitwise=bits_equal(radii, pair[1]))
                            checks.append(row)
                            assert row['same_ids'] and row['rgb_bitwise'] and row['radii_bitwise'], row
                if args.quality:
                    gt = source.scene.ground_truth(frame).clamp(0, 1)
                    values = {name: metrics(result[0], gt) for name, result in results.items()}
                    base = values['A_fp32_F2']
                    for name, (image, radii, ids, stats) in results.items():
                        pair = 'A_fp32_F2' if name in ['A_fp32_F2','B_fp32_new','C_tensor_best_F2','network_only_F2'] else 'C_tensor_best_F2' if name in ['D_tensor_best_new','H_alternative37'] else 'D_tensor_best_new'
                        overlap = 1. if not len(ids) else float(torch.isin(ids, baseline_ids).float().mean())
                        quality.append(dict(scene=spec['id'], split='tune' if spec['slug'] in TUNE else 'heldout',
                            condition=condition, frame=frame, motion=motion, variant=name, N=source.runtime.total, K=len(ids),
                            baseline_psnr=base['psnr'], baseline_ssim=base['ssim'], baseline_lpips=base['lpips'],
                            psnr_loss=base['psnr']-values[name]['psnr'], same_ids=torch.equal(ids, baseline_ids), id_overlap=overlap,
                            renderer_pair=pair, renderer_psnr_loss=values[pair]['psnr']-values[name]['psnr'],
                            rgb_max_error=float((image-results['A_fp32_F2'][0]).abs().max()), **values[name]))
                    del gt
                del results
            if not args.quality:
                for repeat in range(10):
                    for j, (motion, frame) in enumerate(views):
                        for name in order(names, repeat+j+ci, reverse=args.round == 2):
                            torch.cuda.synchronize()
                            tick = time.perf_counter_ns()
                            image, radii, ids, stats = render(name, frame, condition)
                            torch.cuda.synchronize()
                            elapsed = (time.perf_counter_ns()-tick)*1e-6
                            assert len(ids) == expected[name, frame]
                            samples.append(dict(scene=spec['id'], split='tune' if spec['slug'] in TUNE else 'heldout',
                                condition=condition, frame=frame, motion=motion, variant=name, scope='pipeline',
                                round=args.round, repeat=repeat, N=source.runtime.total, K=len(ids), ms=elapsed))
                for motion, frame in [views[0], views[-1]]:
                    for repeat in range(6):
                        for name in order(names, repeat+ci, reverse=args.round == 2):
                            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            start.record()
                            render(name, frame, condition)
                            end.record()
                            end.synchronize()
                            events.append(dict(scene=spec['id'], condition=condition, frame=frame, variant=name,
                                round=args.round, repeat=repeat, cuda_ms=start.elapsed_time(end)))
            print('STUDENT_JOINT', spec['id'], condition, args.round, args.quality, flush=True)
        del networks, renders, variants, dispatch, source, student
        gc.collect()
        torch.cuda.empty_cache()
    for name, data in [('samples',samples),('events',events),('checks',checks),('quality',quality),('memory',memory)]:
        append(out/(name+'.csv'), data)
    save(out/'IDENTITIES.json', identities)
    save(out/'DONE.json', dict(passed=True, samples=len(samples), checks=len(checks), quality=len(quality), no_images=True))


if __name__ == '__main__':
    main()
