"""固定候选/选点输入、FP32 同工作量的 CUDA Event 与墙钟交错微测。"""
from pathlib import Path
import json
import statistics
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source/python'))
from budget_viewer import BudgetScene
from frozen_inference_cuda import FrozenResidualInference
from anchor_frustum import anchor_frustum_mask, compact_gaussians
from selector import compensated_attributes
from renderer import prepare_render_data
from utils import MiniCam, focal2fov

torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32 = False
scene = BudgetScene('/home/prmi/zwj/realLOD/results/fr1_xyz_frozen_master_20261103_01/r20_40_60_80/training/13052_shutdown',
    '/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz')
model, pc, c = scene.model, scene.gaussians, scene.calibration
camera = MiniCam(c['width'], c['height'], focal2fov(c['fx'], c['width']),
    focal2fov(c['fy'], c['height']), np.linalg.inv(scene.views[0]['pose']))
engine = FrozenResidualInference(model, pc)
engine.decode_fusion = '--fused-decode' in sys.argv
engine.prefix_cache = '--prefix-cache' in sys.argv
visible = anchor_frustum_mask(model.anchors.centers, model.anchors.offset_scales,
    camera.full_proj_transform, scene.renderer.occupied)
rows = torch.nonzero(visible[model.anchors.rows]).flatten()
compact = compact_gaussians(pc, rows)
measurements = []


@torch.no_grad()
def measured(module, ratio, functions):
    for function in functions.values():
        for _ in range(6):function()
    for iteration in range(60):
        names = list(functions)
        if iteration % 2:names.reverse()
        for name in names:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize();wall = time.perf_counter();start.record()
            functions[name]();end.record();end.synchronize()
            measurements.append(dict(module=module, ratio=ratio, backend=name,
                gpu_event_ms=start.elapsed_time(end), wall_ms=1000*(time.perf_counter()-wall)))


@torch.no_grad()
def run():
    for ratio in (.01, .05, .1, .2, .4, .8):
        tensor = pc.get_xyz.new_tensor([[ratio]])
        measured('selector_input_and_mlp', ratio, dict(
            pytorch=lambda:model.forward_selection_features(compact.get_xyz, compact.get_scaling,
                compact.get_features_dc, compact.get_opacity, tensor, source_rows=rows, return_logits=True)[2],
            cuda=lambda:engine.logits(rows, ratio)))
        _, ids, _ = scene.renderer.prepare(camera, ratio_budget=ratio)
        mask = torch.zeros(len(rows), device='cuda', dtype=torch.bool)
        # rows 和 ids 都是递增的；建立与生产 reference 完全一致的 selected_mask。
        mask[torch.searchsorted(rows, ids)] = True
        def original():
            attributes = compensated_attributes(model, compact, mask, ratio, source_rows=rows)
            if attributes is not None:
                attributes['scaling'] = attributes['scaling'].log().exp()
                attributes['rotation'] = pc.rotation_activation(attributes['rotation'])
            result = prepare_render_data(compact, mask, attributes)
            if any(not torch.isfinite(value).all() for value in result['inputs'].values() if value is not None):
                raise RuntimeError('nonfinite')
            return result
        functions = dict(pytorch=original, cuda=lambda:engine.render_data(ids, ratio))
        measured('selected_to_render_inputs', ratio, functions)
        engine.graph = True
        measured('selected_to_render_inputs_graph', ratio,
                 dict(pytorch=original, cuda_graph=lambda:engine.render_data(ids, ratio)))
        engine.graph = False
    summary = []
    for key in sorted({(row['module'], row['ratio'], row['backend']) for row in measurements}):
        values = [row for row in measurements if (row['module'], row['ratio'], row['backend']) == key]
        summary.append(dict(module=key[0], ratio=key[1], backend=key[2],
            event_ms=statistics.median(row['gpu_event_ms'] for row in values),
            wall_ms=statistics.median(row['wall_ms'] for row in values)))
    (ROOT/'MICROBENCHMARK.json').write_text(json.dumps(dict(summary=summary, rows=measurements), indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


run()
