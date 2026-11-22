"""统一已提交GPU相机的准备阶段同步/Event测量，独立于完整CPU相机请求口径。"""
from release_joint import *


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--group', required=True, type=int)
    parser.add_argument('--round', required=True, type=int)
    args = parser.parse_args()
    out = HERE / 'reports' / args.output
    out.mkdir(exist_ok=False)
    mode = json.loads((HERE / 'reports/SELECTED_RASTER_20261115.json').read_text())['mode']
    rows, events = [], []
    specs = student_specs()[args.group * 2:args.group * 2 + 2]
    if args.round == 2:
        specs.reverse()
    for spec in specs:
        source, student, _ = load_case(spec)
        networks, factory, base, raster, dispatch = release_engines(source, student, mode)
        engines = dict(A_fp32=networks['fp32'][0], C_tensor=networks['tensor_best'][0],
                       I_previous=dispatch['I_previous'].network, D_final=dispatch['D_final'].network)
        conditions = CONDITIONS[:-1]
        for condition in conditions:
            views = motions(source)
            if args.round == 2:
                views.reverse()
            for motion, frame in views:
                cam = factory.from_source(source, frame)
                for engine in engines.values():
                    for _ in range(2):
                        p.request(engine, cam, condition)
                for repeat in range(8):
                    for name in order(list(engines), repeat, reverse=args.round == 2):
                        torch.cuda.synchronize()
                        start = time.perf_counter_ns()
                        _, ids, _ = p.request(engines[name], cam, condition)
                        torch.cuda.synchronize()
                        rows.append(dict(scene=spec['id'], condition=condition, frame=frame, motion=motion,
                            variant=name, round=args.round, repeat=repeat, K=len(ids), N=source.runtime.total,
                            scope='prepare_given_GPU_camera', ms=(time.perf_counter_ns() - start) * 1e-6))
                for name in engines:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    p.request(engines[name], cam, condition)
                    end.record()
                    end.synchronize()
                    events.append(dict(scene=spec['id'], condition=condition, frame=frame, variant=name,
                        round=args.round, cuda_ms=start.elapsed_time(end)))
            print('RELEASE_PREPARE', spec['id'], condition, args.round, flush=True)
        del source, student, networks, dispatch, engines, base, raster
        gc.collect()
        torch.cuda.empty_cache()
    append(out / 'samples.csv', rows)
    append(out / 'events.csv', events)
    save(out / 'DONE.json', dict(passed=True, samples=len(rows), scope='prepare_given_GPU_camera'))


if __name__ == '__main__':
    main()
