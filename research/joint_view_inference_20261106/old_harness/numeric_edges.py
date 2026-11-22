"""解码融合的密集 FP32 边界、2/6 个 ratio 节点和 checkpoint 重载检查。"""
from pathlib import Path
import json
import sys
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source/python'))
from budget_viewer import BudgetScene
from frozen_inference_cuda import FrozenResidualInference
from selector import load_selector

torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32 = False
scene = BudgetScene('/home/prmi/zwj/realLOD/results/fr1_xyz_frozen_master_20261103_01/r20_40_60_80/training/13052_shutdown')
model, pc = scene.model, scene.gaussians
engine = FrozenResidualInference(model, pc)
checks = []


def same(label, a, b):
    a, b = a.contiguous(), b.contiguous()
    dtype = {1: torch.uint8, 4: torch.int32, 8: torch.int64}[a.element_size()]
    ok = a.shape == b.shape and torch.equal(a.view(dtype), b.view(dtype))
    checks.append(dict(label=label, bitwise=ok, numel=a.numel()))
    if not ok:
        mismatch = torch.nonzero(a.view(dtype).flatten() != b.view(dtype).flatten()).flatten()
        checks[-1].update(first=mismatch[:10].tolist(), actual=a.flatten()[mismatch[:10]].tolist(),
                          expected=b.flatten()[mismatch[:10]].tolist())
        (ROOT/'NUMERIC_EDGE_VALIDATION.json').write_text(json.dumps(dict(passed=False, checks=checks), indent=2))
        raise AssertionError(checks[-1])


@torch.no_grad()
def run():
    generator = torch.Generator(device='cuda').manual_seed(347)
    n = 100003
    random = lambda shape, lo, hi: lo+(hi-lo)*torch.rand(shape, device='cuda', generator=generator)
    xyz, rawscale = random((n, 3), -100, 100), random((n, 3), -103, 60)
    rawopacity = random((n, 1), -105, 105)
    rotation = torch.nn.functional.normalize(random((n, 4), -1, 1), dim=1)
    dc, rest = random((n, 1, 3), -20, 20), random((n, 15, 3), -.1, .1)
    residual = random((n, 14), -100, 100)
    edge = torch.tensor([-105., -100., -90., -87., -20., -1., -.0, .0, .1, 1.,
                         19.458675, 20., 80., 88., 90., 100.], device='cuda')
    residual[:len(edge), 3:6] = edge[:, None]
    residual[:len(edge), 10:11] = edge[:, None];rawopacity[:len(edge)] = 0
    rows = torch.randperm(n, device='cuda', generator=generator)
    rotation_out = rotation[rows]
    widths = [3, 1, 3, 4, 3, 45, 3]
    for additive in (False, True):
        # 加性分支仅验证合法正 Scale 域，负 Scale 沿原 finite 检查报错。
        delta = residual.clone()
        if additive:delta[:, 3:6].abs_()
        scale = rawscale[rows].exp()
        if additive:scale = scale+delta[:, 3:6]
        else:scale = (scale*torch.nn.functional.softplus(delta[:, 3:6]+0.5413248546129181)).clamp_min(1.1754943508222875e-38)
        expected = [xyz[rows]+delta[:, :3], (rawopacity[rows]+delta[:, 10:11]).sigmoid(),
            scale.log().exp(), rotation_out, dc[rows]+delta[:, 11:14].unsqueeze(1), rest[rows], torch.zeros(n, 3, device='cuda')]
        actual = engine.ext.publish_selected([xyz, rawopacity, rawscale.exp(), rotation, dc, rest],
            rows, delta, rotation_out, not additive)
        start = 0
        for j, value in enumerate(expected):
            count = n*widths[j];same(f'decode/additive{additive}/part{j}', actual[start:start+count].view(value.shape), value);start += count
    # checkpoint 不保存推理缓存；重新加载后从原状态建立相同输入与输出。
    saved = ROOT/'roundtrip_selector.pt';torch.save(model.state_dict(), saved)
    loaded = load_selector(str(saved), 'cuda');other = FrozenResidualInference(loaded, pc)
    ids = torch.arange(73, device='cuda')
    for ratio in (.01, .2, .3, .8, 1.):
        same('checkpoint/logits', engine.logits(ids, ratio), other.logits(ids, ratio))
        a, b = engine.render_data(ids, ratio), other.render_data(ids, ratio)
        for key, value in a['inputs'].items():
            if value is not None:same('checkpoint/'+key, value, b['inputs'][key])
    for knots in ((.2, .8), (.01, .05, .1, .2, .3, .4)):
        a = model.anchors
        a.compensation_values = list(knots)
        a.compensation_ratio_knots = torch.tensor(knots, device='cuda', dtype=torch.float64)
        for j in range(len(knots)):
            setattr(a, 'compensation_features_'+str(j), torch.nn.Parameter(random((len(a.centers), 48), -1., 1.)))
        changed = FrozenResidualInference(model, pc)
        for ratio in sorted(set((.01, .025, .075, .15, .25, .35, .5, .7, 1.)+knots)):
            for k in (1, 73, 1000):
                selected = torch.arange(k, device='cuda')
                owners = a.rows[selected]
                offset = ((torch.asinh((pc.get_xyz[selected]-a.centers[owners])/a.offset_scales[owners])-
                           model.spatial_compensation.mean[3:6])/model.spatial_compensation.stddev[3:6]).clamp(-15, 15)
                expected = torch.cat((offset, a.compensation_lookup(selected, ratio)), 1)
                same(f'knots{len(knots)}/r{ratio}/k{k}', changed.compensation_input(selected, changed.controls(ratio)), expected)
    result = dict(passed=True, comparisons=len(checks), compared_values=sum(row['numel'] for row in checks), checks=checks)
    (ROOT/'NUMERIC_EDGE_VALIDATION.json').write_text(json.dumps(result, indent=2)+'\n')
    print('PASS', result['comparisons'], result['compared_values'], flush=True)


run()
