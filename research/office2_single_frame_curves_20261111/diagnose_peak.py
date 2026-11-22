"""只读定位650–780帧耗时峰值：Tile实例、像素贡献、半径及同选点Master对照。"""
from pathlib import Path
import importlib.util
import json
import hashlib
import csv

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('peak_base', HERE/'benchmark.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
p, h, torch, np = b.p, b.h, b.torch, b.np
FRAMES = [450, 500, 550, 650, 675, 700, 722, 741, 753, 775, 800, 850, 900, 950]


@torch.no_grad()
def main():
    output = HERE/'peak_diagnosis'
    output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    c, d, cm, dm = h.configure()
    before = json.loads((HERE/'reports/PROTECTED_AFTER.json').read_text())
    assert all(h.sha(name) == sha for name, sha in before.items())
    scene, source = p.load_scene(b.prior.SLUG)
    student, path, payload = p.student_for(source, b.prior.SLUG)
    engine = h.engine(source, student, 'D')
    camera = h.Camera(source, 'D', dm, b.SIZE)
    full, all_ids = engine.full()
    spec = importlib.util.spec_from_file_location('fr1xyz_production_counts', b.COUNTER)
    binding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(binding)
    old = list(csv.DictReader((HERE/'reports/QUALITY_COUNTS.csv').open()))
    quality = {(r['method'], int(r['frame'])): r for r in old}
    timing = {(r['method'], int(float(r['frame']))): r
              for r in csv.DictReader((HERE/'reports/PLOT_DATA.csv').open())}
    rows = []

    def record(frame, method, variant, cam, packet, ids, st):
        result = b.counted(binding, cam, packet, source.scene.background)
        radii = result[3].float()
        positive = radii[radii > 0]
        pc = packet['inputs']
        scale = pc['scales'].max(dim=1).values
        means = pc['means3D']
        homogeneous = torch.cat([means, torch.ones_like(means[:, :1])], dim=1)
        depth = (homogeneous@cam.world_view_transform)[:, 2]
        depth = depth[radii > 0]
        counts = result[5]
        ref = quality[method, frame]
        if variant == 'compensated':
            assert hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest() == ref['ids_sha256']
            assert int((radii > 0).sum()) == int(ref['projected_K'])
            assert int((counts > 0).sum()) == int(ref['contributing_K'])
        row = dict(frame=frame, method=method, variant=variant, K=len(ids),
            M=st['candidate_gaussians'], tile_instances=int(result[0]),
            accepted_pixel_gaussian_hits=int(counts.sum()),
            contributing_K=int((counts > 0).sum()),
            radius_mean_px=float(positive.mean()), radius_p95_px=float(torch.quantile(positive, .95)),
            radius_max_px=float(positive.max()),
            max_axis_scale_p50=float(torch.quantile(scale, .5)),
            max_axis_scale_p95=float(torch.quantile(scale, .95)),
            max_axis_scale_max=float(scale.max()),
            projected_depth_p05=float(torch.quantile(depth, .05)),
            projected_depth_p50=float(torch.quantile(depth, .5)),
            original_raster_ms=float(timing[method, frame]['raster_ms']))
        if variant == 'compensated':
            master = full['inputs']['scales'].index_select(0, ids).max(dim=1).values
            ratios = scale/master
            row.update(scale_ratio_p50=float(torch.quantile(ratios, .5)),
                scale_ratio_p95=float(torch.quantile(ratios, .95)), scale_ratio_max=float(ratios.max()))
        else:
            row.update(scale_ratio_p50=1., scale_ratio_p95=1., scale_ratio_max=1.)
        rows.append(row)

    for frame in FRAMES:
        cam = camera.create(frame)
        for method in b.METHODS[1:]:
            request = dict(ratio_budget=.5) if method == 'ratio_50' else dict(budget=int(method[6:]))
            packet, ids, stats = engine.prepare(cam, **request)
            record(frame, method, 'compensated', cam, packet, ids, stats)
            master = dict(sh_degree=full['sh_degree'], inputs={
                name: value.index_select(0, ids) if isinstance(value, torch.Tensor) and value.ndim and len(value) == engine.total else value
                for name, value in full['inputs'].items()})
            record(frame, method, 'same_ids_master', cam, master, ids, stats)
        p.write_csv(output/'PER_FRAME.csv', rows)
        print('PROGRESS', frame, flush=True)
    changed = [name for name, sha in before.items() if h.sha(name) != sha]
    result = dict(passed=not changed, changed=changed, protected_files=len(before),
        frames=FRAMES, records=len(rows), selected_ids_projected_and_contributing_match_previous=True,
        counter='original GI: returned[0]=Gaussian-to-Tile duplicates; sum returned[5]=accepted Alpha pixel-Gaussian interactions',
        same_ids_master='same selected IDs, replace all compensated attributes by uncompensated Master; no retraining',
        new_timing=False, training=False, images_saved=0, production_changed=False)
    p.save(output/'VALIDATION.json', result)
    assert result['passed']
    print('COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
