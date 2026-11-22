"""真实冻结 checkpoint 的分层逐位检查，结果保存在独立研究目录。"""
from pathlib import Path
import json
import sys
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source/python'))
from budget_viewer import BudgetScene
from frozen_inference_cuda import FrozenResidualInference
from selector import compensated_attributes
from anchor_frustum import compact_gaussians
from renderer import prepare_render_data

torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
scene = BudgetScene('/home/prmi/zwj/realLOD/results/fr1_xyz_frozen_master_20261103_01/r20_40_60_80/training/13052_shutdown',
    '/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz')
model, pc = scene.model, scene.gaussians
engine = FrozenResidualInference(model, pc)
engine.decode_fusion = '--fused-decode' in sys.argv
prefix = '--prefix-cache' in sys.argv
if prefix:
    if not hasattr(engine, 'prefix_cache'):raise ValueError('前缀缓存候选已回退')
    engine.prefix_cache = True
RESULT = ROOT/('PREFIX_VALIDATION.json' if prefix else
              ('FUSED_DECODE_VALIDATION.json' if engine.decode_fusion else 'VALIDATION.json'))
checks = []


def same(name, actual, expected):
    a, b = actual.contiguous(), expected.contiguous()
    integer = {1: torch.uint8, 4: torch.int32, 8: torch.int64}[a.element_size()]
    ok = a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(integer), b.view(integer))
    record = dict(name=name, shape=list(a.shape), bitwise=ok)
    if not ok and a.shape == b.shape:
        record.update(max_abs=(a-b).abs().max().item() if a.numel() else 0,
                      different=int((a.view(integer) != b.view(integer)).sum()))
    checks.append(record)
    if not ok:
        RESULT.write_text(json.dumps(dict(passed=False, checks=checks), indent=2))
        raise AssertionError(record)


@torch.no_grad()
def run():
    generator = torch.Generator(device='cuda').manual_seed(918)
    ratios = sorted(set([.01, .025, .05, .075, .1, .15, .2, .25, .3, .35, .4, .5, .6, .7, .8, 1.]+
        [max(1e-5, min(1., r+e)) for r in model.anchors.compensation_values for e in (-2e-8, 0., 2e-8)]))
    n = len(pc.get_xyz)
    for size in (1, 7, 32, 129, 1000, n):
        rows = torch.randperm(n, device='cuda', generator=generator)[:size].sort().values
        compact = compact_gaussians(pc, rows)
        for ratio in ratios:
            label = f'n{size}/r{ratio:.10f}'
            owners = model.anchors.rows[rows]
            reference = model.anchors.offset_scales[owners]
            offset = (compact.get_xyz-model.anchors.centers[owners])/reference
            expected_input = torch.cat((offset, compact.get_scaling, compact.get_features_dc.squeeze(1),
                compact.get_opacity, pc.get_xyz.new_tensor([[ratio]]).log().expand(size, 1)), 1)
            same(label+'/selector_input', engine.selector_input(rows, ratio), expected_input)
            expected_logits = model.forward_selection_features(compact.get_xyz, compact.get_scaling,
                compact.get_features_dc, compact.get_opacity, pc.get_xyz.new_tensor([[ratio]]),
                source_rows=rows, return_logits=True)[2]
            same(label+'/logits', engine.logits(rows, ratio), expected_logits)
            controls = engine.controls(ratio)
            features = model.anchors.compensation_lookup(rows, ratio)
            expected_pack = torch.cat((((torch.asinh(offset)-model.spatial_compensation.mean[3:6])/
                model.spatial_compensation.stddev[3:6]).clamp(-15, 15), features), 1)
            same(label+'/compensation_input', engine.compensation_input(rows, controls), expected_pack)
            residual = model.spatial_compensation(compact.get_xyz, compact._scaling, compact.get_rotation,
                compact.get_features_dc, compact._opacity, model.anchors.centers[owners], reference, ratio,
                log_scale_input=True, anchor_feature=features)
            same(label+'/delta14', engine.residual(rows, controls), residual)
        print('layer checks', size, flush=True)
    for size in (0, 1, 17, 256, 1000, n):
        rows = torch.randperm(n, device='cuda', generator=generator)[:size].sort().values
        mask = torch.zeros(n, dtype=torch.bool, device='cuda');mask[rows] = True
        for ratio in (.01, .1, .2, .3, .4, .7, .8, 1.):
            for compensation in (False, True):
                attributes = compensated_attributes(model, pc, mask, ratio) if compensation else None
                if attributes is not None:
                    attributes['scaling'] = attributes['scaling'].log().exp()
                    attributes['rotation'] = pc.rotation_activation(attributes['rotation'])
                expected = prepare_render_data(pc, mask, attributes)
                actual = engine.render_data(rows, ratio, compensation)
                for key, value in expected['inputs'].items():
                    if value is not None:same(f'render/k{size}/r{ratio}/{compensation}/{key}', actual['inputs'][key], value)
    for n in (0, 1, 17, 1024, 13350):
        scores = torch.randint(-4, 5, (n,), device='cuda', generator=generator).float()
        if n > 2:scores[0] = -0.;scores[1] = 0.
        eligible = torch.rand(n, device='cuda', generator=generator) > .2
        available = torch.nonzero(eligible).flatten()
        for k in sorted(set((0, min(1, len(available)), min(13, len(available)), len(available)))):
            expected = available[scores[available].argsort(descending=True, stable=True)[:k]].sort().values
            same(f'stable_selection/n{n}/k{k}', engine.ext.exact_select(scores, eligible, k), expected)
    engine.graph = True
    for k in (1, 17, 200, 1000, 1700, 2560, 6000, 17):
        for ratio in (.01, .3, .8):
            rows = torch.randperm(len(pc.get_xyz), device='cuda', generator=generator)[:k].sort().values
            engine.graph = False;expected = engine.render_data(rows, ratio)
            engine.graph = True;actual = engine.render_data(rows, ratio)
            for key, value in expected['inputs'].items():
                if value is not None:same(f'graph/k{k}/r{ratio}/{key}', actual['inputs'][key], value)
            saved = {key: value.clone() for key, value in actual['inputs'].items() if value is not None}
            engine.render_data(rows.flip(0).contiguous(), .4)
            for key, value in saved.items():same('graph_output_ownership/'+key, actual['inputs'][key], value)
    result = dict(passed=True, comparisons=len(checks), checks=checks,
        graph_entries=len(engine.graphs), graph_allocated_bytes=engine.graph_bytes,
        torch=torch.__version__, tf32=torch.backends.cuda.matmul.allow_tf32)
    RESULT.write_text(json.dumps(result, indent=2)+'\n')
    print('PASS', len(checks), flush=True)


run()
