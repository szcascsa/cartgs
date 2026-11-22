"""相同59维输入，XYZ/Scale/Rotation/Other四个独立两层补偿头。"""
import copy
import gc
import importlib.util
import json
from pathlib import Path
import torch
from torch import nn
import matrix_runtime as rt

ROOT = Path('/home/prmi/zwj/s00-raw-shared-split-20261026')
OLD = Path('/home/prmi/zwj/selector-capacity-matrix-20261026')
PREVIOUS = Path('/home/prmi/zwj/s00-direct-compensation-20261026')
EXTRA = Path('/home/prmi/zwj/s00-direct-compensation-extra-20261026')
DEPTH = Path('/home/prmi/zwj/s00-raw-shared-depth-20261026')
PROTOCOL = json.loads((ROOT / 'protocol.json').read_text())
assert rt.sha(EXTRA / 'run.py') == PROTOCOL['previous_run_sha256']
spec = importlib.util.spec_from_file_location('previous_raw_shared', EXTRA / 'run.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
rt.ROOT = ROOT
PreviousNetwork = previous.Network


class SplitCompensation(nn.Module):
    GROUPS = [('xyz', 0, 3), ('scale', 3, 6), ('rotation', 6, 10), ('other', 10, 14)]

    def __init__(self, original):
        super().__init__()
        self.branches = nn.ModuleDict()
        for name, start, end in self.GROUPS:
            output = nn.Linear(32, end-start, device='cuda')
            with torch.no_grad():
                output.weight.copy_(original[4].weight[start:end])
                output.bias.copy_(original[4].bias[start:end])
            self.branches[name] = nn.Sequential(copy.deepcopy(original[0]), nn.ReLU(),
                copy.deepcopy(original[2]), nn.ReLU(), output)

    def forward(self, inputs):
        return torch.cat([self.branches[name](inputs) for name, _, _ in self.GROUPS], 1)


class Network(PreviousNetwork):
    def __init__(self, scene, cfg, seed):
        super().__init__(scene, cfg, seed)
        if cfg.get('split_heads'):
            assert cfg.get('attach_shared') and self.compensation[0].in_features == 59
            self.compensation = SplitCompensation(self.compensation)


rt.Network = Network


def gate(scene):
    assert json.loads((ROOT / 'results/fr1_xyz/scene_manifest.json').read_text()) == json.loads((OLD / 'results/fr1_xyz/scene_manifest.json').read_text())
    cfg = PROTOCOL['candidate']
    reference_cfg = dict(cfg, id='RawSharedE16', split_heads=False)
    base = Network(scene, reference_cfg, PROTOCOL['seed'])
    net = Network(scene, cfg, PROTOCOL['seed'])
    for name, value in base.state_dict().items():
        if name.startswith(('encoder.', 'selector.', 'extra_tables.')):
            assert torch.equal(value, net.state_dict()[name]), name
    assert len({branch[0].weight.data_ptr() for branch in net.compensation.branches.values()}) == 4
    assert len({branch[2].weight.data_ptr() for branch in net.compensation.branches.values()}) == 4
    for name, start, end in SplitCompensation.GROUPS:
        branch = net.compensation.branches[name]
        for layer in [0, 2]:
            assert torch.equal(branch[layer].weight, base.compensation[layer].weight)
            assert torch.equal(branch[layer].bias, base.compensation[layer].bias)
        assert torch.equal(branch[4].weight, base.compensation[4].weight[start:end])
        assert torch.equal(branch[4].bias, base.compensation[4].bias[start:end])
    for ratio in rt.RATIOS:
        p, h, _ = net(ratio)
        p0, h0, _ = base(ratio)
        assert torch.equal(p, p0) and torch.equal(h, h0)
        ids = torch.nonzero(p > scene.uniform).flatten()
        delta = net.compensate(h, ids, ratio)
        assert delta.shape == (len(ids), 14) and torch.count_nonzero(delta) == 0
        assert torch.equal(delta, base.compensate(h0, ids, ratio))
    with torch.no_grad():
        base.compensation[-1].weight.fill_(.01)
        for branch in net.compensation.branches.values():
            branch[-1].weight.fill_(.01)
    maximum_error = 0.
    for ratio in rt.RATIOS:
        p, h, _ = net(ratio)
        _, h0, _ = base(ratio)
        ids = torch.nonzero(p > scene.uniform).flatten()
        a, b = net.compensate(h, ids, ratio), base.compensate(h0, ids, ratio)
        assert torch.allclose(a, b, atol=1e-7, rtol=1e-6)
        maximum_error = max(maximum_error, float((a-b).abs().max()))
    net.zero_grad(set_to_none=True)
    p, h, _ = net(.05)
    ids = torch.nonzero(p > scene.uniform).flatten()
    net.compensate(h, ids, .05).square().sum().backward()
    branch_gradients = {}
    for name, branch in net.compensation.branches.items():
        norms = [float(branch[layer].weight.grad.norm()) for layer in [0, 2, 4]]
        assert all(value > 0 for value in norms)
        branch_gradients[name] = norms
    assert net.extra_tables[1].grad is not None and net.extra_tables[1].grad.norm() > 0
    assert all(net.extra_tables[j].grad is None for j in [0, 2, 3])
    assert any(p.grad is not None and p.grad.norm() > 0 for p in net.encoder.parameters())
    assert all(p.grad is None for p in net.selector.parameters())
    rt.save_json(ROOT / 'correctness.json', dict(N=scene.N, M=scene.M, source_manifest_equal=True,
        input_width=59, branch_outputs=[3, 3, 4, 4], independent_branch_hidden_parameters=True,
        initial_probabilities_exact=True, initial_compensation_exact_zero=True,
        common_initial_hidden_weights_exact=True, output_row_order_preserved=True,
        nonzero_projection_probe_max_error=maximum_error, branch_gradient_norms=branch_gradients,
        active_latent_gradient=True, inactive_ratio_gradient_none=True, compensation_encoder_gradient=True,
        unified_compensation_parameters=sum(p.numel() for p in base.compensation.parameters()),
        split_compensation_parameters=sum(p.numel() for p in net.compensation.parameters()),
        gradient_probe='temporary nonzero output projection; not actual zero-output step1'))
    assert scene.unchanged()
    del base, net
    gc.collect()


def main():
    old = json.loads((OLD / 'delivery_verification.json').read_text())
    assert rt.sha(ROOT / 'matrix_runtime.py') == old['prototype_sha256']['runtime.py']
    assert rt.sha(Path(old['renderer_binary'])) == old['renderer_sha256']
    scene = rt.Scene('fr1_xyz')
    gate(scene)
    cfg = PROTOCOL['candidate']
    directory = ROOT / 'results/fr1_xyz' / cfg['id'] / str(PROTOCOL['seed'])
    info = rt.train(scene, cfg, PROTOCOL['seed'], PROTOCOL['steps'], directory, checkpoints=[4000, 8000, 12000])
    net = Network(scene, cfg, PROTOCOL['seed'])
    payload = torch.load(info['checkpoint'], map_location='cuda')
    net.load_state_dict(payload['state_dict'])
    for split in ['validation', 'test']:
        output = directory / (split + '_full.csv')
        if not output.exists():
            rows = rt.evaluate(scene, net.eval(), split, full_metrics=True, policies=('same_k', 'cached'))
            for row in rows:
                row.update(config=cfg['id'], seed=PROTOCOL['seed'])
            rt.write_csv(output, rows)
    assert scene.unchanged()
    rt.save_json(ROOT / 'runs.json', [info])
    rt.save_json(ROOT / 'status.json', dict(stage='TRAINING_COMPLETE', completed=1, total=1))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        (ROOT / 'run.exit').write_text('1')
        raise
    (ROOT / 'run.exit').write_text('0')
