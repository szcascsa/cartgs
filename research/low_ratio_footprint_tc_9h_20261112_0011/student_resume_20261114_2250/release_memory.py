"""当前最终版本单实例部署显存；源地图驻留相同，输出用后释放。"""
from release_joint import *


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--group', required=True, type=int)
    args = parser.parse_args()
    out = HERE / 'reports' / args.output
    out.mkdir(exist_ok=False)
    mode = json.loads((HERE / 'reports/SELECTED_RASTER_20261115.json').read_text())['mode']
    config = dict(reuse=True, fused=True, finish=True)
    rows = []
    for spec in student_specs()[args.group * 2:args.group * 2 + 2]:
        source, student, _ = load_case(spec)
        factory = NativeCameraPreparer(p.MiniCam)
        bg = source.scene.background
        for name in ['C_native_F2', 'I_previous', 'D_final']:
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter_ns()
            if name == 'C_native_F2':
                engine = PoseRenderer(source, student, 0)
            elif name == 'I_previous':
                engine = FinalStudentRenderer(source, student, 182, **config, members=15)
            else:
                engine = JointReleaseRenderer(source, student, mode, **config)
            camera = factory.from_source(source, motions(source, quick=True)[0][1])
            engine.precompile(camera, bg)
            torch.cuda.synchronize()
            startup = (time.perf_counter_ns() - start) * 1e-6
            for condition in ['local0.01', 'local0.1', 'local0.3', 'local0.5', 'full']:
                for _, frame in motions(source, quick=True):
                    result = engine.render_from_source(source, frame, bg, condition, factory)
                    del result
            torch.cuda.synchronize()
            rows.append(dict(scene=spec['id'], variant=name, N=source.runtime.total, startup_ms=startup,
                live_increment_bytes=torch.cuda.memory_allocated() - before,
                peak_increment_bytes=torch.cuda.max_memory_allocated() - before,
                raster_workspace_bytes=engine.native.workspace_bytes(),
                projection_bytes=engine.network._pipeline._preparation_bytes(),
                tensor_bytes=engine.network._pipeline._tensor_bytes(),
                selector_prefix_bytes=engine.network._pipeline._prefix_bytes()))
            del engine, camera
            gc.collect()
            torch.cuda.empty_cache()
        del source, student
        gc.collect()
        torch.cuda.empty_cache()
    append(out / 'memory.csv', rows)
    save(out / 'DONE.json', dict(passed=True, single_instance=True, rows=len(rows),
        startup_scope='已建立CUDA上下文后的模块加载、缓存构建与首次Graph预热；不含源码编译或地图文件读取'))
    print('RELEASE_MEMORY_PASS', len(rows), flush=True)


if __name__ == '__main__':
    main()
