"""真实Student尾部近似开发集：同输入、双轮、质量和交互诊断置于计时外。"""
from student_study import *
from balanced_order import order
from formal_render import candidate_workload
from final_benchmark import native_modules
from diagnostics_check import diagnostics
from runtime import SparseRenderer
from tail_runtime import TailRenderer, MODES as TAIL_MODES


def tail_workload(engine, cam, packet, bg, image, radii):
    if not len(radii):
        return dict(preprocess_visible=0,alpha_participating=0,alpha_interactions=0,tail_pixels=0,
            tail_skipped_members=0,tile_instances=0,active_tiles=0,members_p50=0.,members_p95=0.,
            members_max=0,overflow_tiles=0,workspace_bytes=engine.native.workspace_bytes())
    check, flags, count, tail = engine.native.diagnose()
    assert bits_equal(check,image), ('tail diagnostic parity',engine.mode,float((check-image).abs().max()))
    counts = engine.native.snapshot()[3].cpu().numpy()
    return dict(preprocess_visible=int((radii>0).sum()),alpha_participating=int(flags.sum()),
        alpha_interactions=int(count),tail_pixels=int(tail[0]),tail_skipped_members=int(tail[1]),
        tile_instances=int(counts.sum()),active_tiles=int((counts>0).sum()),tiles=len(counts),
        members_p50=float(np.quantile(counts,.5)),members_p95=float(np.quantile(counts,.95)),
        members_max=int(counts.max()),overflow_tiles=int((counts>256).sum()),
        workspace_bytes=engine.native.workspace_bytes())


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--group', required=True, type=int)
    parser.add_argument('--round', required=True, type=int)
    args = parser.parse_args()
    out = HERE / 'reports' / args.output
    out.mkdir(exist_ok=False)
    frozen = json.loads((previous.PRIOR / 'reports/FROZEN_CANDIDATE.json').read_text())
    native_modules(frozen)
    specs = student_specs(tune=True)
    if args.round == 2:
        specs.reverse()
    samples, events, quality, checks, workloads, memory, identities = [], [], [], [], [], [], []
    if args.round == 1:
        import lpips
        from torchmetrics.functional.image import structural_similarity_index_measure as ssim
        perceptual = lpips.LPIPS(net='alex').cuda().eval()
    for spec in specs:
        source, student, path = load_case(spec)
        identities.append(identity(spec, source, student, path))
        network = network_variants(source, student)['fp32'][0]
        bg = source.scene.background
        views = motions(source, quick=True)
        if args.round == 2:
            views.reverse()
        cam = source.camera(views[0][1])
        engines = {'F2': BaselineRenderer()}
        engines.update({MODES[mode]: Renderer(mode) for mode in [37, 38, 40]})
        engines.update({TAIL_MODES[mode]: TailRenderer(mode) for mode in [38, 41, 42, 43, 44, 45]})
        for name, engine in engines.items():
            start = time.perf_counter_ns()
            engine.precompile(source.runtime.total, cam, bg)
            memory.append(dict(scene=spec['id'], variant=name, round=args.round,
                workspace_bytes=engine.native.workspace_bytes(), startup_ms=(time.perf_counter_ns() - start) * 1e-6))
        diagnostic = SparseRenderer(**frozen['regular']) if args.round == 1 else None
        conditions = ['local0.01','local0.05','local0.1','local0.2','local0.3','local0.4','local0.5','local0.7','full']
        if args.round == 2:
            conditions.reverse()
        for ci, condition in enumerate(conditions):
            cases = []
            for motion, frame in views:
                cam = source.camera(frame)
                packet, ids, stats = p.request(network, cam, condition)
                reference, ref_radii = engines['F2'].render(cam, packet, bg)
                if args.round == 1:
                    gt = source.scene.ground_truth(frame).clamp(0, 1)
                    x = reference.clamp(0, 1)
                    baseline = dict(psnr=float(-10 * torch.log10((x - gt).square().mean().clamp_min(1e-30))),
                        ssim=float(ssim(x[None], gt[None], data_range=1.)),
                        lpips=float(perceptual(x[None] * 2 - 1, gt[None] * 2 - 1)))
                for name, engine in engines.items():
                    image, radii = engine.render(cam, packet, bg)
                    assert bits_equal(radii, ref_radii) and bool(torch.isfinite(image).all())
                    checks.append(dict(scene=spec['id'], condition=condition, frame=frame, variant=name, round=args.round,
                        K=len(ids), radii_bitwise=True, rgb_bitwise=bits_equal(image, reference),
                        max_rgb_error=float((image - reference).abs().max())))
                    if args.round == 1 and name != 'F2':
                        x = image.clamp(0, 1)
                        psnr = float(-10 * torch.log10((x - gt).square().mean().clamp_min(1e-30)))
                        quality.append(dict(scene=spec['id'], split='tune' if spec['slug'] in TUNE else 'heldout',
                            condition=condition, frame=frame, motion=motion, variant=name, N=source.runtime.total, K=len(ids),
                            baseline_psnr=baseline['psnr'], psnr=psnr, psnr_loss=baseline['psnr'] - psnr,
                            baseline_ssim=baseline['ssim'], ssim=float(ssim(x[None], gt[None], data_range=1.)),
                            baseline_lpips=baseline['lpips'], lpips=float(perceptual(x[None] * 2 - 1, gt[None] * 2 - 1))))
                        if engine.mode >= 37:
                            values = tail_workload(engine,cam,packet,bg,image,radii) if engine.mode>=41 else candidate_workload(engine,cam,packet,bg,image,radii)
                            workloads.append(dict(scene=spec['id'], condition=condition, frame=frame, variant=name,
                                N=source.runtime.total, K=len(ids), **values))
                    for _ in range(3):
                        engine.render(cam, packet, bg)
                if args.round == 1:
                    values = diagnostics(diagnostic, cam, packet, bg)
                    workloads.append(dict(scene=spec['id'], condition=condition, frame=frame, variant='F2',
                        N=source.runtime.total, K=len(ids), **{key: value for key, value in values.items()
                        if key not in ['K', 'width', 'height', 'bbox_instances', 'depth_words', 'nonempty_words_mean']}))
                cases.append((motion, frame, cam, packet, len(ids)))
            for repeat in range(12):
                for index, (motion, frame, cam, packet, count) in enumerate(cases):
                    for name in order(list(engines), repeat + index + ci, reverse=args.round == 2):
                        torch.cuda.synchronize()
                        start = time.perf_counter_ns()
                        engines[name].render(cam, packet, bg)
                        torch.cuda.synchronize()
                        samples.append(dict(scene=spec['id'], condition=condition, frame=frame, motion=motion,
                            variant=name, round=args.round, repeat=repeat, N=source.runtime.total, K=count,
                            ms=(time.perf_counter_ns() - start) * 1e-6))
            for _, frame, cam, packet, count in [cases[0], cases[-1]]:
                for repeat in range(10):
                    for name in order(list(engines), repeat + ci, reverse=args.round == 2):
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        engines[name].render(cam, packet, bg)
                        end.record()
                        end.synchronize()
                        events.append(dict(scene=spec['id'], condition=condition, frame=frame, variant=name, round=args.round,
                            repeat=repeat, K=count, cuda_ms=start.elapsed_time(end)))
            print('STUDENT_TAIL_DEVELOPMENT', spec['id'], condition, args.round, flush=True)
        del source, student, network, engines, diagnostic, cases
        gc.collect()
        torch.cuda.empty_cache()
    for name, rows in [('samples', samples), ('events', events), ('quality', quality), ('checks', checks),
                       ('workloads', workloads), ('memory', memory)]:
        if name == 'workloads':
            import pandas as pd
            pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
        else:
            append(out / (name + '.csv'), rows)
    save(out / 'IDENTITIES.json', identities)
    save(out / 'DONE.json', dict(passed=True, samples=len(samples), quality_rows=len(quality), no_images=True))


if __name__ == '__main__':
    main()
