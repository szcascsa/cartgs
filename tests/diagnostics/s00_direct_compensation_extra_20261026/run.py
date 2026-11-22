"""fr1_xyz单次追加：Raw11+latent32，Raw11+latent16+h32。"""
import gc
import json
import math
from pathlib import Path
import torch
from torch import nn
import matrix_runtime as rt

ROOT = Path('/home/prmi/zwj/s00-direct-compensation-extra-20261026')
OLD = Path('/home/prmi/zwj/selector-capacity-matrix-20261026')
PREVIOUS = Path('/home/prmi/zwj/s00-direct-compensation-20261026')
PROTOCOL = json.loads((ROOT / 'protocol.json').read_text())
BaseNetwork = rt.Network
rt.ROOT = ROOT


class Network(BaseNetwork):
    def __init__(self, scene, cfg, seed):
        # 原矩阵的初始化reservoir仅支持28列；先建16维临时头，再替换为实际输入。
        bootstrap = dict(cfg, extra=min(cfg['extra'], 16)) if 'compensation_input' in cfg else cfg
        super().__init__(scene, bootstrap, seed)
        self.cfg = cfg
        if 'compensation_input' not in cfg:
            return
        extra = cfg['extra']
        if extra != bootstrap['extra']:
            self.extra_tables = nn.ParameterList([nn.Parameter(torch.zeros(scene.M, extra, device='cuda')) for _ in rt.TRAIN])
        raw_width = 3 if cfg['compensation_input'] == 'offset' else 11
        shared_width = 32 if cfg.get('attach_shared', False) else 0
        first = nn.Linear(raw_width + extra + shared_width, 32, device='cuda')
        common = nn.Linear(11, 32, device='cuda')
        rt.linear_init(common, seed + 300)
        generator = torch.Generator(device='cuda').manual_seed(seed + 305)
        columns = torch.rand((32, 28), generator=generator, device='cuda') * .1 - .05
        if extra > 28:
            columns = torch.cat([columns, torch.rand((32, extra-28), generator=generator, device='cuda')*.1-.05], 1)
        with torch.no_grad():
            first.weight[:, :raw_width].copy_(common.weight[:, :raw_width])
            first.bias.copy_(common.bias)
            first.weight[:, raw_width:raw_width+extra].copy_(columns[:, :extra])
            if shared_width:
                shared_generator = torch.Generator(device='cuda').manual_seed(seed + 306)
                first.weight[:, raw_width+extra:].copy_(torch.rand((32, 32), generator=shared_generator, device='cuda')*.1-.05)
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
        inputs = [raw, latent]
        if self.cfg.get('attach_shared', False):
            inputs.append(h[ids])
        return self.compensation(torch.cat(inputs, 1))


rt.Network = Network


def gate(scene):
    assert json.loads((OLD / 'results/fr1_xyz/scene_manifest.json').read_text()) == json.loads((ROOT / 'results/fr1_xyz/scene_manifest.json').read_text())
    reference_cfg = dict(shared=0, ratio=0, extra=16, selector_depth=1, compensation_depth=2,
        id='DirectRawE16', compensation_input='raw11')
    base = Network(scene, reference_cfg, PROTOCOL['seed'])
    checks = []
    for cfg in PROTOCOL['candidates']:
        net = Network(scene, cfg, PROTOCOL['seed'])
        for name, value in base.state_dict().items():
            if name.startswith(('encoder.', 'selector.')) or (name.startswith('compensation.') and not name.startswith('compensation.0.')):
                assert torch.equal(value, net.state_dict()[name]), name
        assert torch.equal(base.compensation[0].weight[:, :27], net.compensation[0].weight[:, :27])
        assert torch.equal(base.compensation[0].bias, net.compensation[0].bias)
        for ratio in rt.RATIOS:
            p, h, _ = net(ratio)
            p0, h0, _ = base(ratio)
            assert torch.equal(p, p0) and torch.equal(h, h0)
            ids = torch.nonzero(p > scene.uniform).flatten()
            assert torch.count_nonzero(net.compensate(h, ids, ratio)) == 0
        with torch.no_grad():
            net.compensation[-1].weight.fill_(.01)
        net.zero_grad(set_to_none=True)
        p, h, _ = net(.05)
        ids = torch.nonzero(p > scene.uniform).flatten()
        net.compensate(h, ids, .05).square().sum().backward()
        encoder_gradient = any(p.grad is not None and p.grad.norm() > 0 for p in net.encoder.parameters())
        assert encoder_gradient == cfg.get('attach_shared', False)
        assert all(p.grad is None for p in net.selector.parameters())
        assert net.extra_tables[1].grad is not None and net.extra_tables[1].grad.norm() > 0
        assert all(net.extra_tables[j].grad is None for j in [0, 2, 3])
        checks.append(dict(config=cfg['id'], input_width=net.compensation[0].in_features,
            initial_probabilities_equal=True, initial_compensation_zero=True, common_raw_and_first16_projection_equal=True,
            compensation_to_encoder_gradient=bool(encoder_gradient), active_ratio_latent_gradient=True,
            inactive_ratio_gradient_none=True, gradient_probe='temporary nonzero final projection, not actual zero-output step1'))
        del net
    rt.save_json(ROOT / 'correctness.json', dict(N=scene.N, M=scene.M, source_manifest_equal=True, checks=checks))
    assert scene.unchanged()
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
        rt.save_json(ROOT / 'status.json', dict(stage='COMPLETE_MODEL', completed=len(runs), total=2, config=cfg['id']))
        print('COMPLETE_MODEL', cfg['id'], flush=True)
        assert scene.unchanged()
        del net, payload
        gc.collect()
        torch.cuda.empty_cache()
    rt.save_json(ROOT / 'status.json', dict(stage='TRAINING_COMPLETE', completed=2, total=2))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        (ROOT / 'run.exit').write_text('1')
        raise
    (ROOT / 'run.exit').write_text('0')
