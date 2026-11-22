"""最终Student真实当前帧Profiler；scope区分准备与完整请求，不混用墙钟和Kernel累计。"""
from release_joint import *


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=int, default=0)
    args = parser.parse_args()
    mode = json.loads((HERE / 'reports/SELECTED_RASTER_20261115.json').read_text())['mode']
    specs = student_specs(tune=True) if args.group == 0 else student_specs()[-2:]
    cases, owners = [], []
    for spec in specs:
        source, student, _ = load_case(spec)
        networks, factory, base, raster, dispatch = release_engines(source, student, mode)
        owners.append((source, student, networks, base, raster, dispatch))
        bg = source.scene.background
        views = motions(source, quick=True)[::3]
        cam = factory.from_source(source, views[0][1])
        for renderer in [base, raster]:
            renderer.precompile(source.runtime.total, cam, bg)
        for engine in dispatch.values():
            engine.precompile(cam, bg)
        for condition in ['local0.01', 'local0.1', 'local0.3', 'local0.5', 'full']:
            for _, frame in views:
                camera = factory.from_source(source, frame)
                for name in ['C_native_F2', 'I_previous', 'new_prepare_182', 'D_final']:
                    for _ in range(3):
                        dispatch[name].render_from_source(source, frame, bg, condition, factory)
                cases.append((spec['slug'], source, frame, condition, factory, bg, networks, dispatch))
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for repeat in range(5):
        for slug, source, frame, condition, factory, bg, networks, dispatch in cases:
            for name in ['C_native_F2', 'I_previous', 'new_prepare_182', 'D_final']:
                torch.cuda.nvtx.range_push('/'.join(['nine_joint', slug, 'student', condition, name, 'pipeline']))
                dispatch[name].render_from_source(source, frame, bg, condition, factory)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
            if condition != 'full':
                cam = factory.from_source(source, frame)
                for name, network in [('A_fp32_prepare', networks['fp32'][0]),
                                      ('C_tensor_prepare', networks['tensor_best'][0]),
                                      ('D_prepare', dispatch['D_final'].network)]:
                    torch.cuda.nvtx.range_push('/'.join(['nine_joint', slug, 'student', condition, name, 'prepare']))
                    p.request(network, cam, condition)
                    torch.cuda.synchronize()
                    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    print('RELEASE_PROFILE_COMPLETE', len(cases), flush=True)


if __name__ == '__main__':
    main()
