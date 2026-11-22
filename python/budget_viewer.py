"""网页优先的固定预算推理服务；保留原生客户端，默认融合CUDA推理。"""
import argparse
import base64
import hashlib
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import struct
import threading
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from anchor_budget import FixedBudgetRenderer
from gaussian_model import GaussianModel
from renderer import prepare_render_data, render_prepared
from selector import load_selector
from utils import MiniCam, focal2fov


def read_config(path):
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():raise ValueError('Cannot read Selector config: ' + str(path))
    def number(key, default):
        node = storage.getNode(key)
        return default if node.empty() else node.real()
    config = dict(selection_mode=storage.getNode('Selector.selection_mode').string(),
        backend=storage.getNode('Selector.backend').string() or 'legacy',
        protection_enabled=bool(number('Selector.protection_enabled', 0)),
        entry_maturity=number('Elastic.entry_maturity', 0.), temperature=number('Selector.temperature', 1.))
    config['mixed_distribution'] = dict(concentration=tuple(int(number('MixedRatio.concentration_'+key, default))
        for key, default in zip(('high', 'middle', 'low'), (8, 12, 20))),
        eta_min=number('MixedRatio.eta_min', .3), eta_max=number('MixedRatio.eta_max', .8))
    config['mixed_seed'] = int(number('MixedRatio.seed', 0))
    storage.release()
    return config


def load_views(checkpoint, dataset):
    """优先使用完整保存轨迹；精确匹配RGB索引，不将自由相机关联到最近RGB。"""
    cameras = json.loads((checkpoint/'cameras.json').read_text())
    first = cameras[0]
    calibration = {k: first[k] for k in ('width', 'height', 'fx', 'fy')}
    trajectory = checkpoint.parent.parent/'CameraTrajectory_TUM.txt'
    views = []
    if dataset and (dataset/'rgb.txt').is_file() and trajectory.is_file():
        rgb = [line.split('#', 1)[0].split() for line in (dataset/'rgb.txt').read_text().splitlines()]
        rgb = [row for row in rgb if row]
        times = np.array([float(row[0]) for row in rgb])
        records = np.atleast_2d(np.loadtxt(trajectory))
        seen = set()
        for row in records:
            index = int(np.abs(times-row[0]).argmin())
            if abs(times[index]-row[0]) > .001 or index in seen:continue
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix()
            pose[:3, 3] = row[1:4]
            image = dataset/rgb[index][1]
            if not image.is_file():raise ValueError('Missing RGB: ' + str(image))
            views.append(dict(name=rgb[index][1], timestamp=float(times[index]),
                              pose=pose.tolist(), image=str(image)))
            seen.add(index)
    if not views:
        for c in cameras:
            pose = np.eye(4);pose[:3, :3] = c['rotation'];pose[:3, 3] = c['position']
            image = dataset/c['img_name'] if dataset else None
            views.append(dict(name=c['img_name'], pose=pose.tolist(),
                              image=str(image) if image and image.is_file() else None))
    return calibration, views


def validate_pose(value):
    pose = np.asarray(value, dtype=np.float64)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all() or
            not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-7) or
            not np.allclose(pose[:3, :3].T@pose[:3, :3], np.eye(3), atol=1e-4) or
            not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-4)):
        raise ValueError('Camera pose must be a finite rigid camera-to-world 4x4 matrix')
    return pose


def image_uri(image, image_format='jpeg'):
    pixels = (image.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)
    extension, parameters = ('.png', [cv2.IMWRITE_PNG_COMPRESSION, 1]) if image_format == 'png' else ('.jpg', [cv2.IMWRITE_JPEG_QUALITY, 95])
    ok, encoded = cv2.imencode(extension, cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), parameters)
    if not ok:raise RuntimeError('Image encoding failed')
    return 'data:image/'+image_format+';base64,' + base64.b64encode(encoded).decode('ascii')


def psnr(image, reference):
    mse = (image.clamp(0, 1)-reference).square().mean().item()
    return '∞' if mse == 0 else -10*np.log10(mse)


def native_frame_packet(image, reference, metadata, started=None, staging=None):
    """本机Viewer使用FP32 RGB原始像素；指标在传输前计算，不经过JPEG量化。"""
    def pixels(tensor):
        return tensor.detach().clamp(0, 1).permute(1, 2, 0).contiguous().cpu().numpy().astype('<f4', copy=False).tobytes()
    pack_started = time.perf_counter()
    packed = staging.display_bytes(image,reference) if staging is not None else None
    image_bytes, reference_bytes = packed if packed is not None else (pixels(image), pixels(reference) if reference is not None else b'')
    header = dict(metadata, width=image.shape[2], height=image.shape[1],
                  image_bytes=len(image_bytes), reference_bytes=len(reference_bytes))
    if started is not None:
        metadata['stats']['native_pack_ms'] = (time.perf_counter()-pack_started)*1000
        metadata['stats']['server_request_ms'] = (time.perf_counter()-started)*1000
    encoded = json.dumps(header, allow_nan=False).encode('utf-8')
    return b''.join((struct.pack('<I', len(encoded)), encoded, image_bytes, reference_bytes))


class BudgetScene:
    def __init__(self, checkpoint, dataset=None, budget=5000, seed=0, config_path=None, inference_backend='cuda'):
        checkpoint = Path(checkpoint).resolve()
        if (checkpoint/'ply').is_dir():checkpoint = checkpoint/'ply'
        if not (checkpoint/'selector.pt').is_file():raise ValueError('Expected a checkpoint ply directory with selector.pt')
        self.checkpoint = checkpoint
        self.dataset = Path(dataset).resolve() if dataset else None
        self.calibration, self.views = load_views(checkpoint, self.dataset)
        self.gaussians = GaussianModel(3)
        candidates = list((checkpoint/'point_cloud').glob('iteration_*/point_cloud.ply'))
        if len(candidates) != 1:raise ValueError('Expected one checkpoint point_cloud.ply')
        self.gaussians.load_ply(str(candidates[0]))
        self.model = load_selector(str(checkpoint/'selector.pt'), 'cuda')
        config = read_config(Path(config_path) if config_path else checkpoint/'selector_config.yaml')
        self.config = config
        self.model.set_backend(config['backend'])
        self.renderer = FixedBudgetRenderer(self.model, self.gaussians, config, seed, 'pytorch')
        self.requested_backend, self.backend_note = inference_backend, ''
        if inference_backend not in ('pytorch', 'cuda', 'cuda_graph'):
            raise ValueError('Unknown inference backend')
        if inference_backend != 'pytorch':
            from frozen_inference_cuda import FrozenResidualInference
            try:
                engine = FrozenResidualInference(self.model, self.gaussians, inference_backend == 'cuda_graph')
            except ValueError as error:
                self.backend_note = str(error)
                logging.warning('Checkpoint不兼容所请求的融合推理，使用PyTorch: %s', error)
            else:
                self.renderer.backend, self.renderer.inference = inference_backend, engine
        self.ranked_renderer = None
        self.full_data = prepare_render_data(self.gaussians)
        from anchor_visibility import AnchorVisibility
        self.anchor_visibility = AnchorVisibility(self.model.anchors)
        self.full_ids = torch.arange(self.renderer.total, device=self.gaussians.get_xyz.device)
        self.background = torch.zeros(3, device='cuda')
        self.default_budget = budget
        self.images = OrderedDict()
        self.lock = threading.Lock()

    def metadata(self):
        return dict(title='Anchor 固定预算 Viewer', checkpoint=str(self.checkpoint),
            total_gaussians=self.renderer.total, total_anchors=len(self.model.anchors.centers),
            anchor_levels=self.anchor_visibility.catalog, inference_backend=self.renderer.backend,
            requested_inference_backend=self.requested_backend, inference_backend_note=self.backend_note,
            gpu_name=torch.cuda.get_device_name(self.gaussians.get_xyz.device),
            trained_ratios=list(self.renderer.knots), default_budget=self.default_budget,
            ranked_lod_defaults=dict(sorting='projection', ratios=[.5, .3, .1], shares=[.2, .3, .5], hysteresis=0.),
            mixed_ratio_defaults=dict(seed=self.config.get('mixed_seed', 0),
                distribution=self.config.get('mixed_distribution', {})),
            view_residual_available=hasattr(self.model,'selector_view_residual') or hasattr(self.model,'selector_view_input_schema'),
            view_conditioning_type='shared_input' if hasattr(self.model,'selector_view_input_schema') else ('residual' if hasattr(self.model,'selector_view_residual') else 'none'),
            calibration=self.calibration, frames=[dict(index=i, name=v['name'], pose=v['pose'],
                has_ground_truth=v['image'] is not None) for i, v in enumerate(self.views)])

    def ground_truth(self, index):
        if index not in self.images:
            path = self.views[index]['image']
            if path is None:return None
            image = cv2.imread(path)
            if image is None:raise ValueError('Cannot read ground truth: ' + path)
            c = self.calibration
            if image.shape[:2] != (c['height'], c['width']):
                raise ValueError('Ground-truth resolution does not match checkpoint camera')
            self.images[index] = torch.as_tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                device='cuda', dtype=torch.float32).permute(2, 0, 1)/255
            if len(self.images) > 8:self.images.popitem(last=False)
        else:self.images.move_to_end(index)
        return self.images[index]

    @torch.no_grad()
    def render(self, request, native=False):
        started = time.perf_counter()
        if not isinstance(request, dict):raise ValueError('Expected a JSON object')
        mode = request.get('mode', 'budget')
        if mode not in ('budget', 'global_ratio', 'full', 'ranked_lod'):raise ValueError('Unknown rendering mode')
        budget = request.get('budget', self.default_budget)
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError('Budget must be a nonnegative integer')
        reference_full = request.get('reference_full', False)
        if not isinstance(reference_full, bool):raise ValueError('reference_full must be boolean')
        image_format = request.get('image_format', 'jpeg')
        if image_format not in ('jpeg', 'png'):raise ValueError('image_format must be jpeg or png')
        if 'pose' in request:
            if 'frame' in request:raise ValueError('Use either a dataset frame or a free pose')
            pose, frame = validate_pose(request['pose']), None
        else:
            frame = request.get('frame', 0)
            if isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < len(self.views):
                raise ValueError('Frame index outside the saved trajectory')
            pose = validate_pose(self.views[frame]['pose'])
        c = self.calibration
        camera = MiniCam(c['width'], c['height'], focal2fov(c['fx'], c['width']),
                         focal2fov(c['fy'], c['height']), np.linalg.inv(pose))
        prepare_started = time.perf_counter()
        from anchor_visibility import filter_render_packet
        self.renderer.check_state()
        measure_inference_stages=request.get('measure_inference_stages',True)
        if not isinstance(measure_inference_stages,bool):raise ValueError('measure_inference_stages must be boolean')
        anchor_mask = self.anchor_visibility.resolve(request.get('anchor_levels'))
        ablation = dict(selection=request.get('selection', 'learned'), compensation=request.get('compensation', True))
        view_conditioned=request.get('view_conditioned',False)
        if not isinstance(view_conditioned,bool):raise ValueError('view_conditioned must be boolean')
        if view_conditioned and mode not in ('global_ratio','full'):ablation['view_conditioned'] = True
        if mode == 'ranked_lod':
            from anchor_rank_lod import RankedAnchorRenderer
            if self.ranked_renderer is None:self.ranked_renderer = RankedAnchorRenderer(self.renderer)
            reset = request.get('lod_reset_history', False)
            if not isinstance(reset, bool):raise ValueError('lod_reset_history must be boolean')
            if reset:self.ranked_renderer.reset()
            mixed = request.get('lod_mixed', False)
            if not isinstance(mixed, bool):raise ValueError('lod_mixed must be boolean')
            mixed_target = request.get('lod_equivalent_ratio', .3) if mixed else None
            if mixed and (isinstance(mixed_target, bool) or not isinstance(mixed_target, (int, float))):
                raise ValueError('lod_equivalent_ratio must be numeric')
            data, ids, stats = self.ranked_renderer.prepare(camera,
                sorting=request.get('lod_sorting', 'projection'), ratios=request.get('lod_ratios', [.5, .3, .1]),
                shares=request.get('lod_shares', [.2, .3, .5]), hysteresis=request.get('lod_hysteresis', 0.),
                colorize=request.get('lod_colorize', False), mixed_target=mixed_target,
                mixed_seed=request.get('lod_mixed_seed', self.config.get('mixed_seed', 0)),
                mixed_sample=request.get('lod_mixed_sample', 0),
                mixed_config=self.config.get('mixed_distribution'), anchor_mask=anchor_mask, **ablation)
        elif mode == 'budget':
            unit = request.get('budget_unit', 'points')
            ablation['measure_time']=measure_inference_stages
            if unit == 'points':data, ids, stats = self.renderer.prepare(camera, budget, anchor_mask=anchor_mask, **ablation)
            elif unit == 'ratio':data, ids, stats = self.renderer.prepare(camera, ratio_budget=request.get('ratio', .4), anchor_mask=anchor_mask, **ablation)
            else:raise ValueError('budget_unit must be points or ratio')
        elif mode == 'global_ratio':
            ablation['measure_time']=measure_inference_stages
            data, ids, stats = self.renderer.prepare(None, global_ratio=request.get('ratio', self.renderer.knots[0]), **ablation)
            before_filter = len(ids)
            data, ids = filter_render_packet(data, ids, anchor_mask, self.model.anchors.rows)
            stats.update(active_gaussians=len(ids), pre_filter_gaussians=before_filter,
                         hidden_gaussians=before_filter-len(ids), anchor_filter_stage='after_global_selection')
            if anchor_mask is not None:stats['effective_budget'] = len(ids)
        else:
            self.renderer.check_state()
            data, ids = filter_render_packet(self.full_data, self.full_ids, anchor_mask, self.model.anchors.rows)
            stats = dict(total_gaussians=self.renderer.total, candidate_gaussians=len(ids),
                visible_anchors=None, eligible_candidates=None, requested_budget=budget,
                active_gaussians=len(ids), effective_budget=None, ratio=None, budget_unit=None,
                requested_ratio=None, selection_scope='full_master', cache_hit=True, selector_ms=0.,
                selection='full', compensation_enabled=False, ablation_group='Full', budget_calibration_ms=0.,
                compensation_ms=0., prepare_ms=(time.perf_counter()-prepare_started)*1000,
                budget_shortfall=0, ratio_in_training_range=True)
        stats['prepare_ms'] = (time.perf_counter()-prepare_started)*1000
        rendered = render_prepared(camera, data, self.background, measure_time=True, return_radii=True)
        image = rendered['render'].clamp(0, 1)
        summary_started = time.perf_counter()
        stats.update(self.anchor_visibility.frame_stats(self.model.anchors, camera, anchor_mask, ids, rendered['radii'], self.renderer.inference))
        stats['anchor_summary_ms'] = (time.perf_counter()-summary_started)*1000
        stats.update(mode=mode, projected_gaussians=sum(level['projected_gaussians'] for level in stats['anchor_levels']),
                     rasterizer_ms=rendered['rendering_ms'],
                     prepare_plus_render_ms=stats['prepare_ms']+rendered['rendering_ms'],
                     frame=frame, view_kind='dataset' if frame is not None else 'free',
                     ground_truth_psnr=None, full_reference_psnr=None, reference_render_ms=0.)
        quality_started = time.perf_counter()
        reference, reference_label, reference_key = None, '无匹配真值', None
        if frame is not None:
            reference = self.ground_truth(frame)
            if reference is not None:
                stats['ground_truth_psnr'] = psnr(image, reference)
                reference_label = '匹配RGB真值'
                reference_key = 'rgb:'+str(frame)
        if reference_full:
            full_data, _ = filter_render_packet(self.full_data, self.full_ids, anchor_mask, self.model.anchors.rows)
            full = rendered if mode == 'full' else render_prepared(camera, full_data, self.background, measure_time=True)
            stats['reference_render_ms'] = 0. if mode == 'full' else full['rendering_ms']
            stats['full_reference_psnr'] = psnr(image, full['render'].clamp(0, 1))
            if reference is None:
                reference, reference_label = full['render'], '当前视角 Full 参考'+('（筛选Anchor）' if anchor_mask is not None else '')
                filter_key = repr(self.anchor_visibility.last_key).encode() if anchor_mask is not None else b'all'
                reference_key = 'full:'+hashlib.sha256(pose.astype('<f8').tobytes()+filter_key).hexdigest()
        stats['quality_and_reference_ms'] = (time.perf_counter()-quality_started)*1000
        engine = self.renderer.inference
        stats.update(inference_backend=self.renderer.backend,
            gpu_allocated_mb=torch.cuda.memory_allocated()/1024**2,
            gpu_reserved_mb=torch.cuda.memory_reserved()/1024**2,
            graph_count=len(engine.graphs) if engine is not None else 0,
            graph_cache_mb=engine.graph_bytes/1024**2 if engine is not None else 0.)
        result = dict(reference_label=reference_label, pose=pose.tolist(), stats=stats, image_format=image_format,
                      reference_key=reference_key, reference_available=reference is not None,
                      reference_unchanged=False, request_id=request.get('request_id'))
        if mode == 'ranked_lod' and stats['lod_colorize']:
            # PSNR始终来自正常外观；着色仅用于显示，额外光栅化耗时单列。
            colored = dict(data, inputs=dict(data['inputs'], colors_precomp=data['lod_colors'], dc=None, shs=None))
            display = render_prepared(camera, colored, self.background, measure_time=True)
            image = display['render'].clamp(0, 1)
            stats['lod_debug_render_ms'] = display['rendering_ms']
        cache_key = reference_key if native or reference_key is None else reference_key+':'+image_format
        result['reference_cache_key'] = cache_key
        if cache_key is not None and request.get('reference_cache_key') == cache_key:
            result['reference_unchanged'] = True
            reference = None
        if native:
            return native_frame_packet(image, reference, result, started, self.renderer.inference)
        encode_started = time.perf_counter()
        result['image'] = image_uri(image, image_format)
        stats['image_encode_ms'] = (time.perf_counter()-encode_started)*1000
        encode_started = time.perf_counter()
        result['reference'] = image_uri(reference, image_format) if reference is not None else None
        stats['reference_encode_ms'] = (time.perf_counter()-encode_started)*1000
        stats['server_request_ms'] = (time.perf_counter()-started)*1000
        return result


class ViewerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        if str(args[1]) not in ('200',):logging.info(format, *args)

    def send(self, code, content, content_type):
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if path == '/':
            self.send(200, Path(__file__).with_name('budget_viewer.html').read_bytes(), 'text/html; charset=utf-8')
        elif path in ('/budget_viewer.css', '/budget_viewer.js'):
            content_type = 'text/css; charset=utf-8' if path.endswith('.css') else 'text/javascript; charset=utf-8'
            self.send(200, Path(__file__).with_name(path[1:]).read_bytes(), content_type)
        elif path == '/api/meta':
            self.send(200, json.dumps(self.server.scene.metadata()).encode(), 'application/json')
        elif path == '/health':
            self.send(200, b'{"ready":true}', 'application/json')
        else:self.send(404, b'Not found', 'text/plain')

    def do_POST(self):
        if self.path not in ('/api/render', '/api/render-native'):
            return self.send(404, b'Not found', 'text/plain')
        try:
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= 65536:raise ValueError('Invalid request size')
            request = json.loads(self.rfile.read(size))
            with self.server.scene.lock:
                result = (self.server.scene.render(request, native=True) if self.path == '/api/render-native'
                          else self.server.scene.render(request))
            self.send(200, result if isinstance(result, bytes) else json.dumps(result, allow_nan=False).encode(),
                      'application/octet-stream' if isinstance(result, bytes) else 'application/json')
        except (ValueError, TypeError, KeyError) as error:
            self.send(400, json.dumps(dict(error=str(error))).encode(), 'application/json')
        except (BrokenPipeError, ConnectionResetError):pass
        except Exception as error:
            logging.exception('Render failed')
            self.send(500, json.dumps(dict(error=str(error))).encode(), 'application/json')


def main():
    parser = argparse.ArgumentParser(description='无头服务器上的固定点数预算浏览器Viewer')
    parser.add_argument('checkpoint', type=Path, help='保存的shutdown或ply目录')
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--budget', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--inference-backend', choices=('pytorch', 'cuda', 'cuda_graph'), default='cuda')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--state-file', type=Path)
    args = parser.parse_args()
    if args.budget < 0:parser.error('budget must be nonnegative')
    torch.set_num_threads(2)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    scene = BudgetScene(args.checkpoint, args.dataset, args.budget, args.seed, args.config, args.inference_backend)
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    server.daemon_threads = True
    server.scene = scene
    if args.state_file:
        import os
        args.state_file.parent.mkdir(parents=True, exist_ok=True)
        args.state_file.write_text(json.dumps(dict(pid=os.getpid(), host=args.host, port=args.port,
            checkpoint=str(scene.checkpoint), ready=True, default_budget=args.budget), indent=2)+'\n')
    print(f'VIEWER_READY http://{args.host}:{args.port} frames={len(scene.views)} budget={args.budget}', flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__ == '__main__':main()
