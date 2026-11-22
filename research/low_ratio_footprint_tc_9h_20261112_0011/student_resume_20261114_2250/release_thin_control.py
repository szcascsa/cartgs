"""补齐起点最快轻量入口的同输入配对；图像逐位相同则复用既有画质证据。"""
from release_joint import *
from thin_dispatch_runtime import ThinDispatchRenderer
from nine_study import nested_packet


class ThinReleaseRenderer(ThinDispatchRenderer):
    def __init__(self, mode):
        self.mode = mode
        renderer = (TailStrongRenderer(mode) if mode in [301, 302] else
                    TailCutoffRenderer(203) if mode == 203 else EncodedSignRenderer(182))
        self.native = renderer.native
        self._draw = self.native.render
        self._empty = None
        self._fov = None


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--group', required=True, type=int)
    parser.add_argument('--round', required=True, type=int)
    parser.add_argument('--workload', choices=['student_lod', 'nested'], required=True)
    args = parser.parse_args()
    out = HERE / 'reports' / args.output
    out.mkdir(exist_ok=False)
    mode = json.loads((HERE / 'reports/SELECTED_RASTER_20261115.json').read_text())['mode']
    specs = (student_specs() if args.workload == 'student_lod' else SCENES)[args.group * 2:args.group * 2 + 2]
    if args.round == 2:
        specs.reverse()
    samples, events, checks = [], [], []
    for spec in specs:
        if args.workload == 'student_lod':
            source, student, _ = load_case(spec)
            network = network_variants(source, student)['fp32'][0]
        else:
            _, source = load_source(spec['slug'])
            full, full_ids, _ = p.request(source.runtime, None, 'full')
        new = (TailStrongRenderer(mode) if mode in [301, 302] else
               TailCutoffRenderer(203) if mode == 203 else EncodedSignRenderer(182))
        engines = dict(F2=BaselineRenderer(), previous182=EncodedSignRenderer(182),
            thin182=ThinDispatchRenderer(182), new_regular=new, new_thin=ThinReleaseRenderer(mode))
        views = motions(source)
        if args.round == 2:
            views.reverse()
        camera, bg = source.camera(views[0][1]), source.scene.background
        for engine in engines.values():
            engine.precompile(source.runtime.total, camera, bg)
        for ci, condition in enumerate(CONDITIONS if args.round == 1 else CONDITIONS[::-1]):
            cases = []
            for motion, frame in views:
                camera = source.camera(frame)
                if args.workload == 'student_lod':
                    packet, ids, _ = p.request(network, camera, condition)
                else:
                    count = min(len(full_ids), int(condition[5:])) if condition.startswith('fixed') else len(full_ids) if condition == 'full' else int(float(condition[5:]) * len(full_ids))
                    packet, ids = nested_packet(full, (count + .01) / max(1, len(full_ids)))
                results = {name: engine.render(camera, packet, bg) for name, engine in engines.items()}
                for fast, reference in [('thin182', 'previous182'), ('new_thin', 'new_regular')]:
                    image, radii = results[fast]
                    row = dict(scene=spec['id'], workload=args.workload, condition=condition, frame=frame,
                        round=args.round, variant=fast, K=len(ids), rgb_bitwise=bits_equal(image, results[reference][0]),
                        radii_bitwise=bits_equal(radii, results[reference][1]))
                    checks.append(row)
                    assert row['rgb_bitwise'] and row['radii_bitwise'], row
                for engine in engines.values():
                    for _ in range(3):
                        engine.render(camera, packet, bg)
                cases.append((motion, frame, camera, packet, len(ids)))
            for repeat in range(12):
                for index, (motion, frame, camera, packet, count) in enumerate(cases):
                    for name in order(list(engines), repeat + index + ci, reverse=args.round == 2):
                        torch.cuda.synchronize()
                        start = time.perf_counter_ns()
                        engines[name].render(camera, packet, bg)
                        torch.cuda.synchronize()
                        samples.append(dict(scene=spec['id'], workload=args.workload, condition=condition,
                            frame=frame, motion=motion, round=args.round, repeat=repeat, variant=name,
                            N=source.runtime.total, K=count, ms=(time.perf_counter_ns() - start) * 1e-6))
            for _, frame, camera, packet, count in [cases[0], cases[-1]]:
                for repeat in range(8):
                    for name in order(list(engines), repeat + ci, reverse=args.round == 2):
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        engines[name].render(camera, packet, bg)
                        end.record()
                        end.synchronize()
                        events.append(dict(scene=spec['id'], workload=args.workload, condition=condition,
                            frame=frame, round=args.round, repeat=repeat, variant=name, K=count, cuda_ms=start.elapsed_time(end)))
            print('RELEASE_THIN', spec['id'], args.workload, condition, args.round, flush=True)
        del source, engines, new, results, cases
        if args.workload == 'student_lod':
            del student, network
        else:
            del full, full_ids
        gc.collect()
        torch.cuda.empty_cache()
    for name, data in [('samples', samples), ('events', events), ('checks', checks)]:
        append(out / (name + '.csv'), data)
    save(out / 'DONE.json', dict(passed=True, samples=len(samples), checks=len(checks),
        no_images=True, historical_thin_reuse=True))


if __name__ == '__main__':
    main()
