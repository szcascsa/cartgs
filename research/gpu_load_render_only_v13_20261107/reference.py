"""复用上一轮的冻结模型、原版CaRtGS渲染和固定视角质量核查。"""
import importlib.util
import os
from pathlib import Path
import time

from common import *
from control import BudgetController, DUTIES, LEVELS, STAGE_SECONDS
from activity import ActivityMonitor


FRAME = 396
METHODS = ('cartgs_full', 'ours_fixed4000', 'ours_adaptive')
QUEUE = Path('/home/prmi/zwj/realLOD/saved_experiments/dual_view_li_best_full_20261107')


def load_module(name, path, package=False):
    spec = importlib.util.spec_from_file_location(name, path,
        submodule_search_locations=[str(Path(path).parent)] if package else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def atomic_text(path, content):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(content)
    temporary.replace(path)


def queue_paused():
    state = json.loads((QUEUE / 'LOD_PIPELINE.json').read_text())
    return (QUEUE / 'HOLD').exists() and state.get('phase') in ('held', 'paused')


def wait_quiet(monitor, status, allow_background=False):
    while True:
        if monitor.error or monitor.process.poll() is not None:
            raise RuntimeError(monitor.error or 'GPU activity monitor exited while waiting')
        ready = len(monitor.samples) >= 3 if allow_background else monitor.quiet()
        if queue_paused() and ready:
            return
        status.update(state='waiting_queue_and_gpu_telemetry' if allow_background else 'waiting_queue_and_quiet_gpu',
            queue_paused=queue_paused())
        save(ROOT / 'STATUS.json', status)
        time.sleep(1)


def load_runtime():
    import deployment_runtime
    import frozen_inference_cuda
    native = load_module('mycartgs_deployment_fp32_v13', V13_BINARY)
    frozen = load_module('mycartgs_frozen_inference_v3', FROZEN_BINARY)
    deployment_runtime.deployment_extension = lambda: native
    frozen_inference_cuda.extension = lambda: frozen
    return deployment_runtime.DeploymentRenderer.from_checkpoint(
        CHECKPOINT, 0, shrinking_radix=True, compensation_tile=16, selector_tiled=True,
    )


class Renderer:
    def __init__(self):
        scene = BudgetScene(CHECKPOINT, DATASET, 4000, 0, None, 'pytorch')
        self.runtime = load_runtime()
        self.camera = camera_from_view(scene.calibration, scene.views[FRAME])
        self.gt = scene.ground_truth(FRAME)
        self.bg = scene.background
        calibration, views = load_views(VANILLA, DATASET)
        assert calibration == scene.calibration
        assert views[FRAME]['timestamp'] == scene.views[FRAME]['timestamp']
        self.cart_camera = camera_from_view(calibration, views[FRAME])
        self.cart_pc = GaussianModel(3)
        ply = list((VANILLA / 'point_cloud').glob('iteration_*/point_cloud.ply'))
        assert len(ply) == 1
        self.cart_pc.load_ply(str(ply[0]))
        self.cart_raster = load_module('original_raster', ROOT / 'source/original_raster/__init__.py', True)
        self.cart_renderer = load_module('original_renderer', ROOT / 'source/original_renderer.py')
        self.cart_renderer.GaussianRasterizationSettings = self.cart_raster.GaussianRasterizationSettings
        self.cart_renderer.GaussianRasterizer = self.cart_raster.GaussianRasterizer
        assert self.runtime.total == 13298 and len(self.cart_pc.get_xyz) == 35301

    def call(self, method, budget=4000):
        torch.cuda.synchronize()
        start = time.perf_counter()
        if method == 'cartgs_full':
            prepared = start
            output = self.cart_renderer.render(self.cart_camera, self.cart_pc, self.bg)
            k = 35301
            stats = {}
        else:
            packet, ids, stats = self.runtime.prepare(self.camera, budget=budget)
            prepared = time.perf_counter()
            output = render_prepared(self.camera, packet, self.bg)
            k = len(ids)
        torch.cuda.synchronize()
        end = time.perf_counter()
        if method != 'cartgs_full':
            assert k == min(budget, stats['eligible_candidates'])
            assert stats['device_counts'] and stats['shrinking_radix']
        return dict(
            K=k, candidate_N=stats.get('candidate_gaussians', 35301),
            eligible_N=stats.get('eligible_candidates', 35301),
            network_ratio=stats.get('ratio'), requested_K=budget if method != 'cartgs_full' else 35301,
            prepare_ms=1000 * (prepared - start), render_ms=1000 * (end - prepared),
            total_ms=1000 * (end - start),
        ), output['render']

    def quality(self, folder):
        rows = []
        signatures = {}
        for method, budget in [('cartgs_full', 35301)] + [('ours', k) for k in LEVELS]:
            record, image = self.call(method, budget)
            reference = image.clone()
            _, repeated = self.call(method, budget)
            assert torch.equal(reference.view(torch.uint8), repeated.view(torch.uint8))
            assert torch.isfinite(reference).all()
            psnr = float(-10 * torch.log10((reference.clamp(0, 1) - self.gt).square().mean()))
            signature = hashlib.sha256(reference.cpu().numpy().tobytes()).hexdigest()
            signatures[method, budget] = signature
            rows.append(dict(method=method, budget=budget, K=record['K'], psnr=psnr,
                image_sha256=signature, repeated_image_bitwise=True))
        write_csv(folder / 'QUALITY.csv', rows)
        return signatures

