"""fr1_xyz单seed：补偿绕过共享编码；仅在隔离实验中使用。"""
import gc
import json
import math
from pathlib import Path
import torch
from torch import nn
import matrix_runtime as rt

ROOT = Path('/home/prmi/zwj/s00-direct-compensation-20261026')
OLD = Path('/home/prmi/zwj/selector-capacity-matrix-20261026')
PROTOCOL = json.loads((ROOT / 'protocol.json').read_text())
BaseNetwork = rt.Network
rt.ROOT = ROOT


class Network(BaseNetwork):
    def __init__(self, scene, cfg, seed):
        super().__init__(scene, cfg, seed)
        if 'compensation_input' not in cfg:
            return
        raw_width = 3 if cfg['compensation_input'] == 'offset' else 11
        first = nn.Linear(raw_width + cfg['extra'], 32, device='cuda')
        common = nn.Linear(11, 32, device='cuda')
        rt.linear_init(common, seed + 300)
        generator = torch.Generator(device='cuda').manual_seed(seed + 305)
        columns = torch.rand((32, 28), generator=generator, device='cuda') * .1 - .05
        with torch.no_grad():
            first.weight[:, :raw_width].copy_(common.weight[:, :raw_width])
            first.bias.copy_(common.bias)
            first.weight[:, raw_width:].copy_(columns[:, :cfg['extra']])
        second = nn.Linear(32, 32, device='cuda')
        rt.linear_init(second, seed + 310)
        last = nn.Linear(32, 14, device='cuda')
        with torch.no_grad():
            last.weight.zero_()
            last.bias.zero_()
        self.compensation = nn.Sequential(first, nn.ReLU(), second, nn.ReLU(), last)

    def compensate(self, h, ids, ratio):
        if 'compensation_input' not in self.cfg:
            return super().compensate(h, ids, ratio)
        raw = self.scene.attrs[ids]
        if self.cfg['compensation_input'] == 'offset':
            raw = raw[:, :3]
        else:
            raw = torch.cat([raw, torch.full((len(ids), 1), math.log(ratio), device=raw.device)], 1)
        latent = self.query(self.extra_tables, ratio, self.scene.owners[ids])
        return self.compensation(torch.cat([raw, latent], 1))


rt.Network = Network


def gate(scene):
    original = json.loads((OLD / 'results/fr1_xyz/scene_manifest.json').read_text())
    assert original == json.loads((ROOT / 'results/fr1_xyz/scene_manifest.json').read_text())
    baseline_cfg = dict(shared=0, ratio=0, extra=8, selector_depth=1, compensation_depth=2, id='S00R00E08D12')
    base = Network(scene, baseline_cfg, PROTOCOL['seed'])
    checks = []
    for cfg in PROTOCOL['candidates']:
        net = Network(scene, cfg, PROTOCOL['seed'])
        for name, value in base.state_dict().items():
            if name.startswith(('encoder.', 'selector.')):
                assert torch.equal(value, net.state_dict()[name]), name
            if name.startswith('compensation.') and not name.startswith('compensation.0.'):
                assert torch.equal(value, net.state_dict()[name]), name
        for ratio in rt.RATIOS:
            p, h, _ = net(ratio)
            p0, h0, _ = base(ratio)
            assert torch.equal(p, p0) and torch.equal(h, h0)
            ids = torch.nonzero(p > scene.uniform).flatten()
            assert torch.equal(net.compensate(h, ids, ratio), base.compensate(h0, ids, ratio))
        with torch.no_grad():
            net.compensation[-1].weight.fill_(.01)
        net.zero_grad(set_to_none=True)
        p, h, _ = net(.05)
        ids = torch.nonzero(p > scene.uniform).flatten()
        net.compensate(h, ids, .05).square().sum().backward()
        assert all(parameter.grad is None for parameter in net.encoder.parameters())
        assert all(parameter.grad is None for parameter in net.selector.parameters())
        assert net.extra_tables[1].grad is not None and net.extra_tables[1].grad.norm() > 0
        assert all(net.extra_tables[i].grad is None for i in [0, 2, 3])
        checks.append(dict(config=cfg['id'], identical_initial_selection=True, initial_delta_exact_zero=True,
            compensation_has_no_encoder_gradient=True, active_latent_gradient=True,
            inactive_ratio_gradient_none=True, input_width=net.compensation[0].in_features))
        del net
    assert scene.unchanged()
    rt.save_json(ROOT / 'correctness.json', dict(N=scene.N, M=scene.M, source_manifest_equal=True, checks=checks))
    del base
    gc.collect()


def main():
    old = json.loads((OLD / 'delivery_verification.json').read_text())
    assert rt.sha(ROOT / 'matrix_runtime.py') == old['prototype_sha256']['runtime.py']
    assert rt.sha(Path(old['renderer_binary'])) == old['renderer_sha256']
    scene = rt.Scene('fr1_xyz')
    gate(scene)
    runs = []
    for cfg in PROTOCOL['candidates']:
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
        runs.append(info)
        rt.save_json(ROOT / 'runs.json', runs)
        rt.save_json(ROOT / 'status.json', dict(stage='COMPLETE_MODEL', completed=len(runs), total=4, config=cfg['id']))
        print('COMPLETE_MODEL', cfg['id'], flush=True)
        assert scene.unchanged()
        del net, payload
        gc.collect()
        torch.cuda.empty_cache()
    rt.save_json(ROOT / 'status.json', dict(stage='TRAINING_COMPLETE', completed=4, total=4))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        (ROOT / 'run.exit').write_text('1')
        raise
    (ROOT / 'run.exit').write_text('0')
