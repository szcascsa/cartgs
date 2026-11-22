"""最终纯Rasterizer代表Profile，真实Student和固定属性嵌套输入单独标注。"""
from release_joint import *
from nine_study import nested_packet


@torch.no_grad()
def main():
    mode = json.loads((HERE / 'reports/SELECTED_RASTER_20261115.json').read_text())['mode']
    cases, owners = [], []
    for spec in student_specs(tune=True):
        source, student, _ = load_case(spec)
        network = network_variants(source, student)['fp32'][0]
        engines = dict(F2=BaselineRenderer(), previous182=EncodedSignRenderer(182),
            selected=TailStrongRenderer(mode) if mode in [301, 302] else TailCutoffRenderer(203) if mode == 203 else EncodedSignRenderer(182))
        owners.append((source, student, network, engines))
        full, _, _ = p.request(source.runtime, None, 'full')
        for workload in ['student_lod', 'nested']:
            for condition in ['local0.01', 'local0.1', 'local0.3', 'local0.5', 'full']:
                for _, frame in motions(source, quick=True)[::3]:
                    cam, bg = source.camera(frame), source.scene.background
                    if workload == 'student_lod':
                        packet, _, _ = p.request(network, cam, condition)
                    else:
                        packet, _ = nested_packet(full, 1. if condition == 'full' else float(condition[5:]))
                    for engine in engines.values():
                        engine.precompile(source.runtime.total, cam, bg)
                        for _ in range(3):
                            engine.render(cam, packet, bg)
                    cases.append((spec['slug'], workload, condition, cam, packet, bg, engines))
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for repeat in range(6):
        for slug, workload, condition, cam, packet, bg, engines in cases:
            for name in order(list(engines), repeat):
                torch.cuda.nvtx.range_push('/'.join(['nine_render', slug, workload, condition, name]))
                engines[name].render(cam, packet, bg)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    print('RELEASE_RASTER_PROFILE', len(cases), flush=True)


if __name__ == '__main__':
    main()
