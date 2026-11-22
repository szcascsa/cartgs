"""先复现四Head并比较等参数统一80头，再固定结构加入基础Rotation。"""
import copy
import gc
import importlib.util
import json
import math
from pathlib import Path
import sys
import torch
from torch import nn
import matrix_runtime as rt

ROOT = Path('/home/prmi/zwj/s00-split-wide-rotation-20261026')
OLD = Path('/home/prmi/zwj/selector-capacity-matrix-20261026')
EXTRA = Path('/home/prmi/zwj/s00-direct-compensation-extra-20261026')
SPLIT = Path('/home/prmi/zwj/s00-raw-shared-split-20261026')
PROTOCOL = json.loads((ROOT / 'protocol.json').read_text())
assert rt.sha(EXTRA / 'run.py') == PROTOCOL['previous_run_sha256']
spec = importlib.util.spec_from_file_location('previous_raw_shared', EXTRA / 'run.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
rt.ROOT = ROOT
PreviousNetwork = previous.Network
GROUPS = [('xyz', 0, 3), ('scale', 3, 6), ('rotation', 6, 10), ('other', 10, 14)]


def canonical_rotation(q):
    q = torch.nn.functional.normalize(q, dim=1)
    pivot = q.gather(1, q.abs().argmax(1, keepdim=True))
    return q * torch.where(pivot < 0, -torch.ones_like(pivot), torch.ones_like(pivot))


class SplitHead(nn.Module):
    def __init__(self, original, rotation=False):
        super().__init__()
        self.rotation_input = rotation
        self.branches = nn.ModuleDict()
        for name, start, end in GROUPS:
            first = copy.deepcopy(original[0])
            if rotation and name != 'other':
                first = nn.Linear(63, 32, device='cuda')
                with torch.no_grad():
                    first.weight[:, :59].copy_(original[0].weight)
                    first.weight[:, 59:].zero_()
                    first.bias.copy_(original[0].bias)
            last = nn.Linear(32, end-start, device='cuda')
            with torch.no_grad():
                last.weight.copy_(original[4].weight[start:end])
                last.bias.copy_(original[4].bias[start:end])
            self.branches[name] = nn.Sequential(first, nn.ReLU(), copy.deepcopy(original[2]), nn.ReLU(), last)

    def forward(self, inputs, rotation=None):
        return torch.cat([self.branches[name](torch.cat([inputs, rotation], 1)
            if self.rotation_input and name != 'other' else inputs) for name, _, _ in GROUPS], 1)


def wide_head(original, seed):
    first = nn.Linear(59, 80, device='cuda')
    second = nn.Linear(80, 80, device='cuda')
    last = nn.Linear(80, 14, device='cuda')
    rt.linear_init(first, seed+330)
    rt.linear_init(second, seed+340)
    with torch.no_grad():
        first.weight[:32].copy_(original[0].weight)
        first.bias[:32].copy_(original[0].bias)
        second.weight[:32, :32].copy_(original[2].weight)
        second.weight[:32, 32:].zero_()
        second.bias[:32].copy_(original[2].bias)
        last.weight.zero_()
        last.bias.copy_(original[4].bias)
        last.weight[:, :32].copy_(original[4].weight)
    return nn.Sequential(first, nn.ReLU(), second, nn.ReLU(), last)


class Network(PreviousNetwork):
    def __init__(self, scene, cfg, seed):
        super().__init__(scene, cfg, seed)
        kind = cfg.get('head_kind', 'unified32')
        if kind == 'unified80':
            self.compensation = wide_head(self.compensation, seed)
        elif kind in ['split4', 'split4_rotation']:
            self.compensation = SplitHead(self.compensation, rotation=kind == 'split4_rotation')
        if kind == 'split4_rotation':
            # 不注册为可训练参数或改变Master；每次仅gather选中K行。
            self.base_rotation = canonical_rotation(scene.g.get_rotation.detach()).detach()

    def compensate(self, h, ids, ratio):
        if self.cfg.get('head_kind') != 'split4_rotation':
            return super().compensate(h, ids, ratio)
        raw = torch.cat([self.scene.attrs[ids], torch.full((len(ids), 1), math.log(ratio), device='cuda')], 1)
        latent = self.query(self.extra_tables, ratio, self.scene.owners[ids])
        inputs = torch.cat([raw, latent, h[ids]], 1)
        return self.compensation(inputs, self.base_rotation[ids])


rt.Network = Network


def config(kind):
    return dict(id=PROTOCOL['configs'][kind], shared=0, ratio=0, extra=16, selector_depth=1,
        compensation_depth=2, compensation_input='raw11', attach_shared=True, head_kind=kind)


def source_gate(scene):
    old = json.loads((OLD / 'delivery_verification.json').read_text())
    assert rt.sha(ROOT / 'matrix_runtime.py') == old['prototype_sha256']['runtime.py']
    assert rt.sha(Path(old['renderer_binary'])) == old['renderer_sha256']
    assert json.loads((ROOT / 'results/fr1_xyz/scene_manifest.json').read_text()) == json.loads((OLD / 'results/fr1_xyz/scene_manifest.json').read_text())


def gate(scene, kind, seed):
    base = Network(scene, config('unified32'), seed)
    net = Network(scene, config(kind), seed)
    for name, value in base.state_dict().items():
        if name.startswith(('encoder.', 'selector.', 'extra_tables.')):
            assert torch.equal(value, net.state_dict()[name]), name
    for ratio in rt.RATIOS:
        p, h, _ = net(ratio)
        p0, h0, _ = base(ratio)
        ids = torch.nonzero(p > scene.uniform).flatten()
        assert torch.equal(p, p0) and torch.equal(h, h0)
        assert torch.equal(net.compensate(h, ids, ratio), base.compensate(h0, ids, ratio))
        assert torch.count_nonzero(net.compensate(h, ids, ratio)) == 0
    if kind == 'split4_rotation':
        assert torch.equal(net.base_rotation, canonical_rotation(-scene.g.get_rotation.detach()))
        assert torch.allclose(net.base_rotation.norm(dim=1), torch.ones(scene.N, device='cuda'), atol=1e-6)
    with torch.no_grad():
        base.compensation[-1].weight.fill_(.01)
        if isinstance(net.compensation, SplitHead):
            for branch in net.compensation.branches.values():
                branch[-1].weight.fill_(.01)
        elif kind == 'unified80':
            net.compensation[-1].weight[:, :32].fill_(.01)
        else:
            net.compensation[-1].weight.fill_(.01)
    max_error = 0.
    for ratio in rt.RATIOS:
        p, h, _ = net(ratio)
        _, h0, _ = base(ratio)
        ids = torch.nonzero(p > scene.uniform).flatten()
        a, b = net.compensate(h, ids, ratio), base.compensate(h0, ids, ratio)
        assert torch.allclose(a, b, atol=2e-6, rtol=2e-6)
        max_error = max(max_error, float((a-b).abs().max()))
    net.zero_grad(set_to_none=True)
    p, h, _ = net(.05)
    ids = torch.nonzero(p > scene.uniform).flatten()
    net.compensate(h, ids, .05).square().sum().backward()
    assert net.extra_tables[1].grad.norm() > 0
    assert all(net.extra_tables[j].grad is None for j in [0, 2, 3])
    assert any(p.grad is not None and p.grad.norm() > 0 for p in net.encoder.parameters())
    assert all(p.grad is None for p in net.selector.parameters())
    rotation_gradients = {}
    if kind == 'split4_rotation':
        for name in ['xyz', 'scale', 'rotation']:
            norm = float(net.compensation.branches[name][0].weight.grad[:, 59:].norm())
            assert norm > 0
            rotation_gradients[name] = norm
        assert net.compensation.branches['other'][0].in_features == 59
    result = dict(kind=kind, seed=seed, source_manifest_equal=True, initial_probability_exact=True,
        initial_compensation_exact=True, initial_nonzero_output_probe_max_error=max_error,
        active_ratio_gradient=True, inactive_ratio_gradient_none=True, compensation_encoder_gradient=True,
        compensation_parameters=sum(p.numel() for p in net.compensation.parameters()),
        geometry_rotation_column_gradients=rotation_gradients,
        gradient_probe='temporary nonzero final output projection; not actual step1')
    rt.save_json(ROOT / f'gate_{kind}_{seed}.json', result)
    assert scene.unchanged()
    del net, base
    gc.collect()


def train_one(scene, kind, seed):
    gate(scene, kind, seed)
    cfg = config(kind)
    directory = ROOT / 'results/fr1_xyz' / cfg['id'] / str(seed)
    info = rt.train(scene, cfg, seed, 12000, directory, checkpoints=[4000, 8000, 12000])
    net = Network(scene, cfg, seed)
    payload = torch.load(info['checkpoint'], map_location='cuda')
    net.load_state_dict(payload['state_dict'])
    for split in ['validation', 'test']:
        output = directory / (split + '_full.csv')
        if not output.exists():
            rows = rt.evaluate(scene, net.eval(), split, full_metrics=True, policies=('same_k', 'cached'))
            for row in rows:
                row.update(config=cfg['id'], seed=seed)
            rt.write_csv(output, rows)
    assert scene.unchanged()
    del net, payload
    gc.collect()
    torch.cuda.empty_cache()
    return info


def main():
    phase = sys.argv[1]
    assert phase in ['A', 'B']
    if phase == 'B':
        assert (ROOT / 'phase_A.exit').read_text().strip() == '0'
    scene = rt.Scene('fr1_xyz')
    source_gate(scene)
    jobs = PROTOCOL['phase_A'] if phase == 'A' else PROTOCOL['phase_B']
    runs = []
    for kind, seed in jobs:
        runs.append(train_one(scene, kind, seed))
        rt.save_json(ROOT / f'runs_{phase}.json', runs)
        rt.save_json(ROOT / 'status.json', dict(stage='PHASE_'+phase, completed=len(runs), total=len(jobs), kind=kind, seed=seed))
        print('COMPLETE', phase, kind, seed, flush=True)
    rt.save_json(ROOT / 'status.json', dict(stage='PHASE_'+phase+'_COMPLETE', completed=len(runs), total=len(jobs)))


if __name__ == '__main__':
    phase = sys.argv[1]
    try:
        main()
    except Exception:
        (ROOT / f'phase_{phase}.exit').write_text('1')
        raise
    (ROOT / f'phase_{phase}.exit').write_text('0')
