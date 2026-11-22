"""同一CaRtGS地图配对比较原版/Mode40；与Ours同轮纯渲染复测。"""
from pathlib import Path
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent/'office2_dynamic04_mode40_20261112/with_ours_full'
spec = importlib.util.spec_from_file_location('cartgs_m40_prior', PRIOR/'benchmark.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
h, p, torch, np, b = previous.h, previous.p, previous.torch, previous.np, previous.b
import cv2
import diff_gaussian_rasterization

METHODS = ['cartgs_original', 'cartgs_mode40', 'ours_full', 'dynamic_04']
SIZE = (2064, 2208)


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    name = 'smoke' if args.smoke else 'formal'
    output = ROOT/name
    output.mkdir(exist_ok=False)
    lock = (output/'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save = lambda key, value: p.save(output/key, value)
    write = lambda key, value: p.write_csv(output/key, value)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    _, config, _, dm = h.configure()
    old = json.loads((PRIOR/'formal/PROTECTED_AFTER.json').read_text())
    before, drift, missing = {}, [], []
    for filename, expected in old.items():
        if Path(filename).is_file():
            current = h.sha(filename)
            before[filename] = current
            if current != expected:
                drift.append(filename)
        else:
            missing.append(filename)
    save('HISTORICAL_MANIFEST_DIFFERENCES.json', dict(changed_before_run=drift, unavailable_before_run=missing))
    binary = b.MODE40/'build/mycartgs_footprint_ninehour_v1.so'
    binary_sha = '4de72de81cbfa1eaaee6d065ce643c57ee875be29b34c690891baf389232656e'
    original_sha = b.prior.b.prior.ORIGINAL_SHA
    assert h.sha(binary) == binary_sha
    assert h.sha(diff_gaussian_rasterization._C.__file__) == original_sha
    spec = importlib.util.spec_from_file_location('cartgs_shared_mode40', b.MODE40/'render_runtime.py')
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    runtime.extension = lambda: h.verified('mycartgs_footprint_ninehour_v1', binary, binary_sha)
    scene, source = p.load_scene('Replica__office2__rgbd')
    student, student_path, _ = p.student_for(source, 'Replica__office2__rgbd')
    assert h.sha(student_path) == 'fd85285c6bd5871f4b41c85690597ccadc965d5a5ec6f474a983e4e09f547a75'
    engine = h.engine(source, student, 'D')
    camera = h.Camera(source, 'D', dm, SIZE)
    master, all_ids = engine.full()
    vanilla = b.prior.b.prior.VANILLA
    ply, = list((vanilla/'model/15900_shutdown/ply/point_cloud').glob('iteration_*/point_cloud.ply'))
    pc = b.prior.b.prior.GaussianModel(3)
    pc.load_ply(str(ply), 'cuda')
    for attribute in ['_xyz', '_scaling', '_rotation', '_opacity', '_features_dc', '_features_rest']:
        getattr(pc, attribute).requires_grad_(False)
    assert len(source.scene.views) == 2000 and engine.total == 43197 and len(pc.get_xyz) == 84203
    records = {float(row[0]): row for row in np.atleast_2d(np.loadtxt(vanilla/'model/CameraTrajectory_TUM.txt'))}
    poses = {}
    for frame, view in enumerate(source.scene.views):
        row = records[float(view['timestamp'])]
        pose = np.eye(4)
        pose[:3, :3] = b.prior.b.Rotation.from_quat(row[4:8]).as_matrix()
        pose[:3, 3] = row[1:4]
        poses[frame] = pose
    frozen_path = PRIOR.parent/'formal/BUDGETS_FROZEN.csv'
    assert h.sha(frozen_path) == 'c18dfb71b1c98505ab159490b024a4cd8dbc9639bb211c1d38eddb19a3d04c8c'
    frozen = {int(row['frame']): row for row in read(frozen_path)}
    expected = {(row['method'], int(row['frame'])): row for row in read(PRIOR/'formal/QUALITY_COUNTS.csv')}
    renderers = {key: runtime.Renderer(40) for key in ['cartgs_mode40', 'ours']}
    for path in [binary, Path(diff_gaussian_rasterization._C.__file__), frozen_path, ply, Path(student_path),
                 vanilla/'model/CameraTrajectory_TUM.txt', PRIOR/'formal/QUALITY_COUNTS.csv',
                 Path(__file__), ROOT/'job_runner.py']:
        before[str(path)] = h.sha(path)
    save('PROTECTED_BEFORE.json', before)
    frames = [0, 500, 650, 700, 722, 741, 1000, 1500, 1900, 1999] if args.smoke else list(range(2000))
    save('PROTOCOL.json', dict(dataset='Replica/office2/rgbd', width=2064, height=2208, frames=len(frames),
        methods=METHODS, repeats=1 if args.smoke else 3, renders_per_request=1,
        cartgs='same 84203-Gaussian PLY and own archived SLAM poses, same packets for original/Mode40',
        ours='same 43197-Gaussian Master, distilled student, Tensor D and new preparation, frozen dynamic K',
        renderer='Mode40 for cartgs_mode40/ours_full/dynamic_04; original for cartgs_original',
        workspace='separate bounded Mode40 instances for CaRtGS and Ours to prevent capacity contamination',
        quality='actual outputs vs INTER_LINEAR resized original RGB, paired RGB errors and radii',
        timing='single raster call CUDA Event plus synchronized host wall; no preparation/network/diagnose/GT/I/O',
        acceptance='200-frame whole-block retry if external GPU or >10% process CPU activity within expanded +/-2s window',
        orders='forward/reverse/seed0-shuffled, rotated method order, all 2000 frames retained',
        inference_only=True, training=False, production_changed=False, saved_render_images=0,
        gpu=torch.cuda.get_device_name(), frozen_budget_sha256=h.sha(frozen_path)))
    save('MODEL_IDENTITIES.json', dict(scene=scene, vanilla_ply=str(ply), vanilla_ply_sha256=h.sha(ply),
        vanilla_N=84203, ours_N=43197, student=str(student_path), student_sha256=h.sha(student_path),
        original_binary=str(diff_gaussian_rasterization._C.__file__), original_sha256=original_sha,
        Mode40_binary=str(binary), Mode40_sha256=binary_sha, D_binary=config['binary'], D_sha256=config['binary_sha256']))

    def prepare(frame, method):
        if method.startswith('cartgs_'):
            return (p.MiniCam(*SIZE, camera.fx, camera.fy, np.linalg.inv(poses[frame])),
                    b.prior.b.renderer.prepare_render_data(pc), None, dict(K=84203, M=84203, ratio=1.))
        cam = camera.create(frame)
        if method == 'ours_full':
            return cam, master, all_ids, dict(K=43197, M=43197, ratio=1.)
        packet, ids, stats = engine.prepare(cam, budget=int(frozen[frame]['K']))
        assert len(ids) == int(frozen[frame]['K']) and stats['candidate_gaussians'] == int(frozen[frame]['M'])
        assert hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest() == frozen[frame]['ids_sha256']
        return cam, packet, ids, dict(K=len(ids), M=stats['candidate_gaussians'], ratio=stats['ratio'])

    def raster(cam, packet, method):
        if method == 'cartgs_original':
            out = p.render_prepared(cam, packet, source.scene.background, return_radii=True)
            return out['render'], out['radii']
        return renderers['cartgs_mode40' if method == 'cartgs_mode40' else 'ours'].render(cam, packet, source.scene.background)

    quality, errors = [], []
    for index, frame in enumerate(frames):
        truth = cv2.imread(source.scene.views[frame]['image'], cv2.IMREAD_COLOR)
        assert truth is not None
        truth = cv2.cvtColor(cv2.resize(truth, SIZE, interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        gt = torch.from_numpy(truth).to('cuda').permute(2, 0, 1)/255.
        cam, packet, _, stats = prepare(frame, 'cartgs_original')
        reference, reference_radii = raster(cam, packet, 'cartgs_original')
        reference = reference.clone(); reference_radii = reference_radii.clone()
        reference_psnr = float(-10*torch.log10((reference.clamp(0, 1)-gt).square().mean()))
        assert reference_psnr == float(expected['cartgs_full', frame]['psnr'])
        for method in METHODS:
            if method == 'cartgs_original':
                image, radii, digest = reference, reference_radii, 'all_map'
                contributed = int(expected['cartgs_full', frame]['contributing_K'])
                interactions = -1
            else:
                if method != 'cartgs_mode40':
                    cam, packet, ids, stats = prepare(frame, method)
                image, radii = raster(cam, packet, method)
                image = image.clone(); radii = radii.clone()
                renderer = renderers['cartgs_mode40' if method == 'cartgs_mode40' else 'ours']
                debug, flags, visits = renderer.native.diagnose()
                assert p.bits_equal(debug, image), (frame, method, 'diagnostic differs')
                contributed, interactions = int((flags != 0).sum()), int(visits[0])
                digest = 'all_map' if method != 'dynamic_04' else hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
            assert image.shape == (3, SIZE[1], SIZE[0]) and torch.isfinite(image).all()
            psnr = float(-10*torch.log10((image.clamp(0, 1)-gt).square().mean()))
            projected = int((radii > 0).sum())
            assert 0 <= contributed <= projected <= stats['K']
            if method in ['ours_full', 'dynamic_04']:
                assert psnr == float(expected[method, frame]['psnr']), (frame, method)
                assert contributed == int(expected[method, frame]['contributing_K'])
            if method == 'cartgs_mode40':
                assert torch.equal(radii, reference_radii)
                error = (image-reference).abs()
                errors.append(dict(frame=frame, original_psnr=reference_psnr, Mode40_psnr=psnr,
                    delta_psnr=psnr-reference_psnr, rgb_abs_max=float(error.max()), rgb_abs_mean=float(error.mean()),
                    radii_bitwise=True, same_packet=True))
            quality.append(dict(frame=frame, method=method, **stats, projected_K=projected, contributing_K=contributed,
                psnr=psnr, interactions=interactions, ids_sha256=digest))
        if index % 100 == 0 or index == len(frames)-1:
            write('QUALITY_COUNTS.csv', quality); write('CARTGS_RENDERER_DIFFERENCES.csv', errors)
            print('QUALITY', index+1, len(frames), flush=True)
    monitor = ROOT/'logs'/(name+'_MONITOR.jsonl')
    attempts, rows, all_rows = [], [], []
    deadline = time.time()+1800

    def observations():
        return [json.loads(line) for line in monitor.read_text().rsplit('\n', 1)[0].splitlines() if line.strip()]

    def quiet():
        while True:
            obs = observations()
            if len(obs) >= 3 and all(not x['foreign_active'] and not x['foreign_cpu_active'] for x in obs[-3:]):
                return
            assert time.time() < deadline, '外部负载等待超过30分钟，保留现场'
            save('STATUS.json', dict(state='waiting_external_load', accepted_blocks=sum(x['accepted'] for x in attempts)))
            time.sleep(2)

    quiet()
    for frame in np.linspace(0, 1999, 3 if args.smoke else 24, dtype=int):
        for method in METHODS:
            cam, packet, _, _ = prepare(int(frame), method)
            raster(cam, packet, method)
    torch.cuda.synchronize()
    for repeat in range(1 if args.smoke else 3):
        order = list(frames)
        if repeat == 1: order.reverse()
        if repeat == 2: np.random.default_rng(0).shuffle(order)
        for block, lo in enumerate(range(0, len(order), 200)):
            while True:
                quiet(); current = []; start = time.time()
                for index in range(lo, min(lo+200, len(order))):
                    frame = order[index]
                    for method in p.rotate(METHODS, repeat+index):
                        cam, packet, _, stats = prepare(frame, method)
                        torch.cuda.synchronize()
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        tick = time.perf_counter_ns(); begin.record()
                        result = raster(cam, packet, method)
                        end.record(); end.synchronize()
                        current.append(dict(frame=frame, method=method, repeat=repeat, **stats,
                            raster_ms=begin.elapsed_time(end), raster_wall_ms=(time.perf_counter_ns()-tick)/1e6))
                finish = time.time()
                while observations()[-1]['time'] < finish+2: time.sleep(.5)
                obs = [x for x in observations() if start-2 <= x['time'] <= finish+2]
                gpu = sum(bool(x['foreign_active']) for x in obs)
                cpu = sum(bool(x['foreign_cpu_active']) for x in obs)
                accepted = len(obs) >= 3 and gpu == cpu == 0
                attempt = dict(repeat=repeat, block=block, attempt=len(attempts), start_unix=start,
                    end_unix=finish, samples=len(current), observations=len(obs), external_gpu_samples=gpu,
                    external_cpu_samples=cpu, accepted=accepted)
                attempts.append(attempt)
                all_rows.extend(dict(row, attempt=attempt['attempt'], accepted=accepted) for row in current)
                write('ALL_TIMINGS.csv', all_rows); save('BLOCK_ATTEMPTS.json', attempts)
                print('BLOCK', json.dumps(attempt), flush=True)
                if accepted:
                    rows.extend(current); write('TIMINGS.csv', rows)
                    break
    after = {filename: h.sha(filename) for filename in before}
    changed = [filename for filename in before if before[filename] != after[filename]]
    save('PROTECTED_AFTER.json', after)
    result = dict(passed=not changed, protected_files=len(before), changed=changed, frames=len(frames),
        methods=METHODS, quality_rows=len(quality), timing_rows=len(rows), original_PSNR_exact=True,
        ours_PSNR_counts_ids_exact=True, Mode40_diagnostic_RGB_bitwise=True, cartgs_radii_bitwise=True,
        frozen_budget_sha256=h.sha(frozen_path), accepted_blocks=sum(x['accepted'] for x in attempts),
        rejected_blocks=sum(not x['accepted'] for x in attempts), training=False, production_changed=False,
        saved_render_images=0, renderer_stats={k:v.native.stats() for k,v in renderers.items()})
    save('VALIDATION.json', result); save('STATUS.json', dict(state='complete', **result))
    assert result['passed']
    print('COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
