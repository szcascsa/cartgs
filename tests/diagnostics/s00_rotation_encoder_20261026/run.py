"""隔离对照：基础Quaternion同时进入共享Encoder和四个补偿Head。"""
import gc
import importlib.util
import json
import math
from pathlib import Path
import torch
from torch import nn
import matrix_runtime as rt

ROOT = Path('/home/prmi/zwj/s00-rotation-encoder-20261026')
PREVIOUS = Path('/home/prmi/zwj/s00-split-wide-rotation-20261026')
PROTOCOL = json.loads((ROOT/'protocol.json').read_text())
assert rt.sha(PREVIOUS/'run.py') == PROTOCOL['previous_run_sha256']
spec = importlib.util.spec_from_file_location('previous_split', PREVIOUS/'run.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
rt.ROOT = ROOT


def config(kind):
    return dict(id=PROTOCOL['configs'][kind], shared=0, ratio=0, extra=16,
        selector_depth=1, compensation_depth=2, compensation_input='raw11',
        attach_shared=True, head_kind='split4', rotation_encoder=kind == 'rotation_all')


class Network(previous.Network):
    def __init__(self, scene, cfg, seed):
        super().__init__(scene, cfg, seed)
        self.rotation_all = cfg['rotation_encoder']
        if not self.rotation_all:
            return
        self.base_rotation = previous.canonical_rotation(scene.g.get_rotation.detach()).detach()
        encoder = nn.Linear(15, 32, device='cuda')
        with torch.no_grad():
            encoder.weight[:, :11].copy_(self.encoder.weight)
            encoder.weight[:, 11:].zero_()
            encoder.bias.copy_(self.encoder.bias)
        self.encoder = encoder
        for branch in self.compensation.branches.values():
            old = branch[0]
            first = nn.Linear(63, 32, device='cuda')
            with torch.no_grad():
                first.weight[:, :11].copy_(old.weight[:, :11])
                first.weight[:, 11:15].zero_()
                first.weight[:, 15:].copy_(old.weight[:, 11:])
                first.bias.copy_(old.bias)
            branch[0] = first

    def raw(self, ratio, ids=None):
        attrs = self.scene.attrs if ids is None else self.scene.attrs[ids]
        raw = torch.cat([attrs, torch.full((len(attrs), 1), math.log(ratio), device=attrs.device)], 1)
        if self.rotation_all:
            q = self.base_rotation if ids is None else self.base_rotation[ids]
            raw = torch.cat([raw, q], 1)
        return raw

    def forward(self, ratio):
        if not self.rotation_all:
            return super().forward(ratio)
        h = torch.relu(self.encoder(self.raw(ratio)))
        logits = self.selector(h)
        p = logits.softmax(-1)[:, 1] * self.scene.eligible
        return p, h, logits[:, 1]-logits[:, 0]

    def compensate(self, h, ids, ratio):
        latent = self.query(self.extra_tables, ratio, self.scene.owners[ids])
        inputs = torch.cat([self.raw(ratio, ids), latent, h[ids]], 1)
        return self.compensation(inputs)


rt.Network = Network


def gate(scene, seed):
    baseline = Network(scene, config('baseline'), seed)
    candidate = Network(scene, config('rotation_all'), seed)
    old = previous.Network(scene, previous.config('split4'), seed)
    assert all(torch.equal(v, old.state_dict()[k]) for k, v in baseline.state_dict().items())
    assert torch.equal(candidate.base_rotation, previous.canonical_rotation(-scene.g.get_rotation.detach()))
    assert candidate.encoder.in_features == 15
    assert all(b[0].in_features == 63 for b in candidate.compensation.branches.values())
    errors = []
    for ratio in rt.RATIOS:
        p0, h0, _ = baseline(ratio)
        p1, h1, _ = candidate(ratio)
        ids = torch.nonzero(p0 > scene.uniform).flatten()
        assert torch.equal(p0, p1)
        assert torch.allclose(h0, h1, atol=2e-6, rtol=2e-6)
        assert torch.count_nonzero(baseline.compensate(h0, ids, ratio)) == 0
        assert torch.equal(baseline.compensate(h0, ids, ratio), candidate.compensate(h1, ids, ratio))
        errors.append(float((h0-h1).abs().max()))
    for net in [baseline, candidate]:
        with torch.no_grad():
            for branch in net.compensation.branches.values():
                branch[-1].weight.fill_(.01)
    p0, h0, _ = baseline(.05)
    p1, h1, _ = candidate(.05)
    ids = torch.nonzero(p0 > scene.uniform).flatten()
    a, b = baseline.compensate(h0, ids, .05), candidate.compensate(h1, ids, .05)
    assert torch.allclose(a, b, atol=2e-6, rtol=2e-6)
    b.square().sum().backward()
    assert float(candidate.encoder.weight.grad[:, 11:].norm()) > 0
    head_gradients = {name:float(branch[0].weight.grad[:, 11:15].norm())
        for name, branch in candidate.compensation.branches.items()}
    assert all(v > 0 for v in head_gradients.values())
    assert candidate.extra_tables[1].grad.norm() > 0
    assert all(candidate.extra_tables[j].grad is None for j in [0, 2, 3])
    probe = Network(scene, config('rotation_all'), seed)
    opt = torch.optim.Adam(probe.parameters(), lr=.001, eps=1e-15, fused=True)
    actual = []
    for step in range(1, 4):
        opt.zero_grad(set_to_none=True)
        p, h, _ = probe(.05)
        selected = torch.nonzero(p.detach() > scene.uniform).flatten()
        delta = probe.compensate(h, selected, .05)
        image, _, _ = scene.render(selected, delta, scene.train[0], 1+p[selected]-p[selected].detach())
        photo = rt.photo(image, scene.train[0]['gt'])
        loss = photo+.01*(p.sum()-.05*scene.N).abs()/(.05*scene.N)
        loss.backward()
        gradients = dict(encoder=float(probe.encoder.weight.grad[:, 11:].norm()),
            **{name:float(branch[0].weight.grad[:, 11:15].norm())
               for name, branch in probe.compensation.branches.items()})
        assert all(math.isfinite(v) for v in gradients.values())
        actual.append(dict(step=step, photo=float(photo), rotation_column_gradient_norms=gradients))
        opt.step()
    assert any(row['rotation_column_gradient_norms']['encoder'] > 0 for row in actual[1:])
    assert all(any(row['rotation_column_gradient_norms'][name] > 0 for row in actual[1:])
        for name in candidate.compensation.branches)
    result = dict(seed=seed, baseline_matches_previous_initial_state=True,
        initial_probabilities_exact=True, initial_compensation_exact=True,
        initial_encoder_max_error=max(errors), nonzero_output_probe_error=float((a-b).abs().max()),
        quaternion_sign_invariant=True, raw_widths=[11,15], head_input_widths=[59,63],
        baseline_head_parameters=sum(p.numel() for p in baseline.compensation.parameters()),
        candidate_head_parameters=sum(p.numel() for p in candidate.compensation.parameters()),
        all_head_rotation_probe_gradients=head_gradients, actual_photo_probe=actual,
        inactive_ratio_gradient_none=True, master_frozen=scene.unchanged())
    assert result['master_frozen']
    rt.save_json(ROOT/f'correctness_{seed}.json', result)
    del baseline, candidate, old, probe, opt
    gc.collect()
    torch.cuda.empty_cache()


def diagnostics(scene, net, directory):
    rows = []
    with torch.no_grad():
        for ratio in rt.RATIOS:
            p, h, logits = net(ratio)
            margin = logits.double()-torch.logit(scene.uniform.double())
            margin[~scene.eligible] = -torch.inf
            ids = torch.argsort(margin, descending=True, stable=True)[:scene.K[ratio]].sort().values
            delta = net.compensate(h, ids, ratio)
            for name, start, end in previous.GROUPS:
                value = delta[:, start:end]
                rows.append(dict(ratio=ratio, group=name, selected=len(ids),
                    rms=float(value.square().mean().sqrt()), max_abs=float(value.abs().max()),
                    p_mean=float(p.mean()), p10=float(p.quantile(.1)), p90=float(p.quantile(.9))))
        stats = dict(encoder_rotation_weight_norm=float(net.encoder.weight[:, 11:].norm()),
            head_rotation_weight_norms={name:float(branch[0].weight[:, 11:15].norm())
                for name, branch in net.compensation.branches.items()}) if net.rotation_all else {}
    rt.write_csv(directory/'compensation_statistics.csv', rows)
    rt.save_json(directory/'rotation_input_updates.json', stats)


def main():
    scene = rt.Scene('fr1_xyz')
    expected = json.loads((PREVIOUS/'results/fr1_xyz/scene_manifest.json').read_text())
    assert json.loads((ROOT/'results/fr1_xyz/scene_manifest.json').read_text()) == expected
    assert rt.sha(ROOT/'matrix_runtime.py') == PROTOCOL['runtime_sha256']
    assert rt.sha(Path(PROTOCOL['renderer_binary'])) == PROTOCOL['renderer_sha256']
    for seed in PROTOCOL['seeds']:
        gate(scene, seed)
    runs = []
    for kind, seed in PROTOCOL['jobs']:
        cfg = config(kind)
        directory = ROOT/'results/fr1_xyz'/cfg['id']/str(seed)
        info = rt.train(scene, cfg, seed, PROTOCOL['steps'], directory, checkpoints=[4000,8000,12000])
        net = Network(scene, cfg, seed)
        payload = torch.load(info['checkpoint'], map_location='cuda')
        net.load_state_dict(payload['state_dict'])
        for split in ['bank','validation','test']:
            target = directory/(split+'_full.csv')
            if not target.exists():
                rows = rt.evaluate(scene, net.eval(), split, full_metrics=True, policies=('same_k','cached'))
                for row in rows:
                    row.update(config=cfg['id'], seed=seed)
                rt.write_csv(target, rows)
        diagnostics(scene, net, directory)
        assert scene.unchanged()
        runs.append(info)
        rt.save_json(ROOT/'runs.json', runs)
        print('COMPLETE', kind, seed, flush=True)
        del net, payload
        gc.collect()
        torch.cuda.empty_cache()
    rt.save_json(ROOT/'status.json', dict(stage='COMPLETE', completed=len(runs), total=len(PROTOCOL['jobs'])))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        (ROOT/'run.exit').write_text('1')
        raise
    (ROOT/'run.exit').write_text('0')
