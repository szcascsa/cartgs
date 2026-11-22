"""仅匹配Student的当前帧NVTX；Nsight分解与独占墙钟分开。"""
from student_study import *
from pose_runtime import PoseRenderer
from pending_runtime import PendingRenderer
from register_runtime import StudentRegisterRenderer
from balanced_order import order


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--register', action='store_true')
    args = parser.parse_args()
    cases, owners = [], []
    for spec in student_specs(tune=True):
        source, student, _ = load_case(spec)
        owners.append((source, student))
        networks = network_variants(source, student)
        factory = networks['fp32'][1]
        renders = {'F2': BaselineRenderer(), 'new': Renderer(37)}
        variants = benchmark_variants(networks, renders)
        dispatch = {'F_native': PendingRenderer(source, student, 37, native_dispatch=True),
                    'G_cpp_pose': PoseRenderer(source, student, 37)}
        if args.register:
            dispatch.update({f'R{w}': StudentRegisterRenderer(source, student, 37, 7, w) for w in [1, 2, 4]})
        views = motions(source, quick=True)[::3]
        cam = factory.from_source(source, views[0][1])
        bg = source.scene.background
        for engine in dispatch.values():
            engine.precompile(cam, bg)
        for engine in renders.values():
            engine.precompile(source.runtime.total, cam, bg)
        for condition in ['local0.01', 'local0.1', 'local0.3', 'local0.5', 'full']:
            for _, frame in views:
                for name, ((engine, _), renderer) in variants.items():
                    for _ in range(3):
                        cam = factory.from_source(source, frame)
                        packet, _, _ = p.request(engine, cam, condition)
                        renderer.render(cam, packet, bg)
                for name, engine in dispatch.items():
                    for _ in range(3):
                        if name == 'F_native':
                            engine.render(factory.from_source(source, frame), bg, condition)
                        else:
                            engine.render_from_source(source, frame, bg, condition, factory)
                cases.append((spec['slug'], condition, source, frame, factory, variants, dispatch))
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for repeat in range(5):
        for index, (slug, condition, source, frame, factory, variants, dispatch) in enumerate(cases):
            for name in order(list(variants) + list(dispatch), repeat + index, reverse=repeat % 2 == 1):
                torch.cuda.nvtx.range_push('/'.join(['nine_joint', slug, 'student', condition, name, 'pipeline']))
                if name in dispatch:
                    if name == 'F_native':
                        dispatch[name].render(factory.from_source(source, frame), source.scene.background, condition)
                    else:
                        dispatch[name].render_from_source(source, frame, source.scene.background, condition, factory)
                else:
                    (engine, camera), renderer = variants[name]
                    cam = camera.from_source(source, frame)
                    packet, _, _ = p.request(engine, cam, condition)
                    renderer.render(cam, packet, source.scene.background)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    print('STUDENT_PROFILE_COMPLETE', len(cases), flush=True)


if __name__ == '__main__':
    main()
