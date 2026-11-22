"""冻结推理缓存的拓扑、状态失效、非默认 Stream 与大 N 边界验证。"""
from pathlib import Path
import json
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source/python'))
from budget_viewer import BudgetScene, read_config
from anchor_budget import FixedBudgetRenderer
from anchor_frustum import anchor_frustum_mask
from utils import MiniCam, focal2fov

torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
scene = BudgetScene('/home/prmi/zwj/realLOD/results/fr1_xyz_frozen_master_20261103_01/r20_40_60_80/training/13052_shutdown',
    '/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz')
config = read_config(scene.checkpoint/'selector_config.yaml')
model, pc = scene.model, scene.gaussians
c = scene.calibration
camera = MiniCam(c['width'], c['height'], focal2fov(c['fx'], c['width']),
    focal2fov(c['fy'], c['height']), np.linalg.inv(scene.views[0]['pose']))
reference = scene.renderer
candidate = FixedBudgetRenderer(model, pc, config, backend='cuda_graph')
checks = []
fields = ('_xyz', '_scaling', '_rotation', '_opacity', '_features_dc', '_features_rest')


def equal(label, a, b):
    a, b = a.contiguous(), b.contiguous()
    integer = {1: torch.uint8, 4: torch.int32, 8: torch.int64}[a.element_size()]
    ok = a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(integer), b.view(integer))
    checks.append(dict(label=label, bitwise=ok))
    assert ok, label


def stale(label):
    for renderer in (reference, candidate):
        try:renderer.prepare(camera, budget=5000)
        except RuntimeError as error:assert 'changed' in str(error)
        else:raise AssertionError('Stale cache '+label)
        renderer.rebuild()
    checks.append(dict(label=label+'/invalidation', bitwise=True))


def compare(label):
    for option in (dict(budget=0), dict(budget=1), dict(budget=5000), dict(ratio_budget=.01),
                   dict(ratio_budget=.05), dict(ratio_budget=.4), dict(ratio_budget=.8)):
        for compensation in (False, True):
            a, ids, _ = reference.prepare(camera, compensation=compensation, **option)
            b, other_ids, _ = candidate.prepare(camera, compensation=compensation, **option)
            prefix = f'{label}/{option}/{compensation}'
            equal(prefix+'/IDs', ids, other_ids)
            for name, value in a['inputs'].items():
                if value is not None:equal(prefix+'/'+name, value, b['inputs'][name])
    engine = candidate.inference
    masks = [torch.zeros_like(candidate.occupied), torch.ones_like(candidate.occupied),
             anchor_frustum_mask(model.anchors.centers, model.anchors.offset_scales,
                                 camera.full_proj_transform, candidate.occupied)]
    for i, visible in enumerate(masks):
        rows, v, available = engine.candidates(visible)
        expected = torch.nonzero(visible[model.anchors.rows]).flatten()
        equal(f'{label}/candidates/{i}', rows, expected)
        assert v == int(visible.sum()) and available == int(engine.eligible[expected].sum())
    print('PASS', label, len(pc.get_xyz), flush=True)


@torch.no_grad()
def run():
    compare('initial')
    model.anchors.compensation_features_1.add_(.001)
    stale('Feature Adam version');compare('feature update')
    pc._xyz.add_(pc._xyz.new_tensor([.01, -.02, .03]))
    model.anchors.centers.add_(pc._xyz.new_tensor([.01, -.02, .03]))
    stale('global geometry correction');compare('geometry correction')
    ids = torch.arange(len(pc.get_xyz)-1, -1, -1, device='cuda')
    for name in fields:setattr(pc, name, getattr(pc, name)[ids])
    a = model.anchors
    a.rows = a.rows[ids]
    if hasattr(a, 'slot_inactive'):a.slot_inactive = a.slot_inactive.flatten()[ids]
    stale('Gaussian reorder');compare('Gaussian reorder')
    # Anchor 重排；同时迁移全部用于本后端的 Feature、格子和 owner 身份。
    order = torch.arange(len(a.centers)-1, -1, -1, device='cuda')
    inverse = torch.empty_like(order);inverse[order] = torch.arange(len(order), device='cuda')
    a.rows = inverse[a.rows]
    for name in ('centers', 'offset_scales', 'keys'):setattr(a, name, getattr(a, name)[order])
    a.features = torch.nn.Parameter(a.features[order])
    for j in range(len(a.compensation_values)):
        name = 'compensation_features_'+str(j);setattr(a, name, torch.nn.Parameter(getattr(a, name)[order]))
    stale('Anchor reorder');compare('Anchor reorder')
    # 新生 Anchor 从父 Anchor 迁移 Feature，新 Gaussian 的几何允许不同。
    parent = a.rows[:1]
    new_owner = len(a.centers)
    for name in ('centers', 'offset_scales', 'keys'):
        value = getattr(a, name);setattr(a, name, torch.cat((value, value[parent]), 0))
    a.features = torch.nn.Parameter(torch.cat((a.features, a.features[parent]), 0))
    for j in range(len(a.compensation_values)):
        name = 'compensation_features_'+str(j);value = getattr(a, name)
        setattr(a, name, torch.nn.Parameter(torch.cat((value, value[parent]), 0)))
    for name in fields:
        value = getattr(pc, name);setattr(pc, name, torch.cat((value, value[:2]), 0))
    pc._xyz[-1].add_(.001)
    a.rows = torch.cat((a.rows, a.rows[:1], a.rows.new_tensor([new_owner])))
    if hasattr(a, 'slot_inactive'):a.slot_inactive = torch.cat((a.slot_inactive.flatten(), a.slot_inactive.flatten()[:2]))
    stale('Clone Split Anchor growing');compare('Clone Split Anchor growing')
    keep = torch.arange(0, len(pc.get_xyz), 3, device='cuda')
    for name in fields:setattr(pc, name, getattr(pc, name)[keep])
    a.rows = a.rows[keep]
    if hasattr(a, 'slot_inactive'):a.slot_inactive = a.slot_inactive.flatten()[keep]
    stale('Prune');compare('Prune')
    stream = torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):compare('nondefault stream')
    torch.cuda.current_stream().wait_stream(stream)
    # 真实权重的大 N 合成负载；不把它称作另一个在线 SLAM 场景。
    for name in fields:setattr(pc, name, getattr(pc, name).repeat((24,)+(1,)*(getattr(pc, name).ndim-1)))
    a.rows = a.rows.repeat(24)
    if hasattr(a, 'slot_inactive'):a.slot_inactive = a.slot_inactive.flatten().repeat(24)
    stale('Large N');compare('Large N')
    for name in fields:setattr(pc, name, getattr(pc, name)[:0])
    a.rows = a.rows[:0]
    if hasattr(a, 'slot_inactive'):a.slot_inactive = a.slot_inactive.flatten()[:0]
    stale('empty topology');compare('empty topology')
    assert len(candidate.inference.graphs) <= 6 and candidate.inference.graph_bytes <= 512*1024**2
    (ROOT/'STRESS_VALIDATION.json').write_text(json.dumps(dict(passed=True, comparisons=len(checks), checks=checks), indent=2)+'\n')
    print('PASS', len(checks), flush=True)


run()
