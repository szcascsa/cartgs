"""fr1_xyz单次深度对照：Raw11+latent16+h32，补偿隐藏层2→3。"""
import gc
import importlib.util
import json
from pathlib import Path
import torch
from torch import nn
import matrix_runtime as rt

ROOT = Path('/home/prmi/zwj/s00-raw-shared-depth-20261026')
OLD = Path('/home/prmi/zwj/selector-capacity-matrix-20261026')
PREVIOUS = Path('/home/prmi/zwj/s00-direct-compensation-20261026')
EXTRA = Path('/home/prmi/zwj/s00-direct-compensation-extra-20261026')
PROTOCOL = json.loads((ROOT / 'protocol.json').read_text())
assert rt.sha(EXTRA / 'run.py') == PROTOCOL['previous_run_sha256']
spec = importlib.util.spec_from_file_location('previous_raw_shared', EXTRA / 'run.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
rt.ROOT = ROOT
PreviousNetwork = previous.Network


class Network(PreviousNetwork):
    def __init__(self, scene, cfg, seed):
        bootstrap = dict(cfg, compensation_depth=2) if cfg.get('compensation_depth') == 3 else cfg
        super().__init__(scene, bootstrap, seed)
        self.cfg = cfg
        if cfg.get('compensation_depth') == 3:
            assert cfg.get('attach_shared') and self.compensation[0].in_features == 59
            old = self.compensation
            added = nn.Linear(32, 32, device='cuda')
            with torch.no_grad():
                added.weight.copy_(torch.eye(32, device='cuda'))
                added.bias.zero_()
            self.compensation = nn.Sequential(old[0], old[1], old[2], old[3], added, nn.ReLU(), old[4])


rt.Network = Network


def gate(scene):
    assert json.loads((ROOT / 'results/fr1_xyz/scene_manifest.json').read_text()) == json.loads((OLD / 'results/fr1_xyz/scene_manifest.json').read_text())
    cfg = PROTOCOL['candidate']
    reference_cfg = dict(cfg, id='RawSharedE16', compensation_depth=2)
    base = Network(scene, reference_cfg, PROTOCOL['seed'])
    net = Network(scene, cfg, PROTOCOL['seed'])
    for name, value in base.state_dict().items():
        if name.startswith(('encoder.', 'selector.', 'extra_tables.', 'compensation.0.', 'compensation.2.')):
            assert torch.equal(value, net.state_dict()[name]), name
    assert torch.equal(base.compensation[4].weight, net.compensation[6].weight)
    assert torch.equal(base.compensation[4].bias, net.compensation[6].bias)
    for ratio in rt.RATIOS:
        p, h, _ = net(ratio)
        p0, h0, _ = base(ratio)
        assert torch.equal(p, p0) and torch.equal(h, h0)
        ids = torch.nonzero(p > scene.uniform).flatten()
        assert torch.equal(net.compensate(h, ids, ratio), base.compensate(h0, ids, ratio))
        assert torch.count_nonzero(net.compensate(h, ids, ratio)) == 0
    with torch.no_grad():
        base.compensation[-1].weight.fill_(.01)
        net.compensation[-1].weight.fill_(.01)
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
    assert net.compensation[4].weight.grad.norm() > 0
    assert net.extra_tables[1].grad is not None and net.extra_tables[1].grad.norm() > 0
    assert all(net.extra_tables[j].grad is None for j in [0, 2, 3])
    assert any(p.grad is not None and p.grad.norm() > 0 for p in net.encoder.parameters())
    assert all(p.grad is None for p in net.selector.parameters())
    rt.save_json(ROOT / 'correctness.json', dict(N=scene.N, M=scene.M, source_manifest_equal=True,
        input_width=59, hidden_widths=[32, 32, 32], common_parameters_exact=True,
        initial_probabilities_exact=True, initial_compensation_zero=True,
        added_layer_identity=True, nonzero_projection_probe_max_error=maximum_error,
        added_layer_gradient=True, active_latent_gradient=True, inactive_ratio_gradient_none=True,
        compensation_encoder_gradient=True, gradient_probe='temporary nonzero output layer, not actual step1'))
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
