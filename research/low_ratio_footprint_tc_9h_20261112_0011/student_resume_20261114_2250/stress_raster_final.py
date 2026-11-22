"""合成压力输入的真实配对测时；无GT，不将参考图像PSNR当成真实场景质量门限。"""
from student_study import *
from balanced_order import order
from encoded_sign_runtime import EncodedSignRenderer
from formal_render import candidate_workload
from final_benchmark import native_modules
from nine_study import nested_packet


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--engine', choices=['tail', 'strong', 'half2', 'cub'], required=True)
    parser.add_argument('--mode', type=int, required=True)
    parser.add_argument('--round', type=int, required=True)
    args = parser.parse_args()
    constructors = {}
    from tail_cutoff_runtime import TailCutoffRenderer
    constructors['tail'] = TailCutoffRenderer
    if args.engine == 'strong':
        from tail_strong_runtime import TailStrongRenderer
        constructors['strong'] = TailStrongRenderer
    elif args.engine == 'half2':
        from encoded_half2_runtime import EncodedHalf2Renderer
        constructors['half2'] = EncodedHalf2Renderer
    elif args.engine == 'cub':
        from cub_depth_runtime import CubDepthRenderer
        constructors['cub'] = CubDepthRenderer
    out = HERE / 'reports' / args.output
    out.mkdir(exist_ok=False)
    native_modules(json.loads((previous.PRIOR / 'reports/FROZEN_CANDIDATE.json').read_text()))
    engines = {'F2': BaselineRenderer(), 'previous182': EncodedSignRenderer(182),
               'candidate': constructors[args.engine](args.mode)}
    bg = torch.tensor([.1, .2, .3], device='cuda')
    samples, events, checks, workloads = [], [], [], []
    resolutions = [(97, 65), (641, 479)]
    styles = ['huge', 'thin', 'overlap', 'equal', 'random', 'barely_alpha']
    counts = [0, 1, 16, 128, 512, 1000, 2500, 5000]
    if args.round == 2:
        resolutions.reverse()
        styles.reverse()
        counts.reverse()
    for width, height in resolutions:
        camera = p.MiniCam(width, height, 1.2, 1., np.eye(4))
        for engine in engines.values():
            engine.precompile(5000, camera, bg)
        for style in styles:
            torch.manual_seed(9701 + styles.index(style) if args.round == 1 else 9701 + len(styles) - styles.index(style) - 1)
            full = synthetic_packet(5000, style)
            for count in counts:
                packet, ids = nested_packet(full, (count + .01) / 5000)
                ref, ref_radii = engines['F2'].render(camera, packet, bg)
                base = dict(style=style, width=width, height=height, K=count, round=args.round)
                for name, engine in engines.items():
                    image, radii = engine.render(camera, packet, bg)
                    assert bits_equal(radii, ref_radii) and torch.equal(torch.isfinite(image), torch.isfinite(ref))
                    checks.append(dict(**base, variant=name, rgb_bitwise=bits_equal(image, ref), radii_bitwise=True,
                        reference_image_psnr=float(-10 * torch.log10((image - ref).square().mean().clamp_min(1e-30))),
                        rgb_max_error=float((image - ref).abs().max())))
                    if name != 'F2' and args.round == 1:
                        workloads.append(dict(**base, variant=name,
                            **candidate_workload(engine, camera, packet, bg, image, radii)))
                    for _ in range(4):
                        engine.render(camera, packet, bg)
                for repeat in range(24):
                    for name in order(list(engines), repeat, reverse=args.round == 2):
                        torch.cuda.synchronize()
                        start = time.perf_counter_ns()
                        engines[name].render(camera, packet, bg)
                        torch.cuda.synchronize()
                        samples.append(dict(**base, variant=name, repeat=repeat,
                            ms=(time.perf_counter_ns() - start) * 1e-6))
                for repeat in range(10):
                    for name in order(list(engines), repeat, reverse=args.round == 2):
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        engines[name].render(camera, packet, bg)
                        end.record()
                        end.synchronize()
                        events.append(dict(**base, variant=name, repeat=repeat, cuda_ms=start.elapsed_time(end)))
            print('STRESS_DONE', width, height, style, flush=True)
    for name, rows in [('samples', samples), ('events', events), ('checks', checks), ('workloads', workloads)]:
        append(out / (name + '.csv'), rows)
    save(out / 'DONE.json', dict(passed=True, input='synthetic_stress_real_timing', gt_available=False,
        no_images=True, engine=args.engine, mode=args.mode, samples=len(samples)))


if __name__ == '__main__':
    main()
