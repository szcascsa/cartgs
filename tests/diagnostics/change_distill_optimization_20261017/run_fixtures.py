from pathlib import Path
import subprocess
import json
import torch

D = Path(__file__).resolve().parent
ROOT = Path('/home/prmi/zwj/mycartgs/results/change_distill_ablation/20261017-2038')
fixtures = {'global': ROOT / 'controls/global_change',
            'aabb_4000': ROOT / 'events_default_v2/aabb_only/event_4000_1',
            'aabb_5000': ROOT / 'events_default_v2/aabb_only/event_5000_2'}
results = {}
for name, event in fixtures.items():
    for stage in ['baseline', 'stage_a_grouped']:
        out = D / stage / 'raw' / ('fixture_' + name)
        assert not out.exists(), str(out)
        out.mkdir(parents=True)
        cmd = [str(D / stage / 'benchmark_verify'), str(event), str(out), 'tf', '5', '8192', '4', '0', '0', '0']
        with (out / 'stdout.log').open('w') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    a = torch.jit.load(str(D / 'baseline/raw' / ('fixture_' + name) / 'verification.pt')).state_dict()
    b = torch.jit.load(str(D / 'stage_a_grouped/raw' / ('fixture_' + name) / 'verification.pt')).state_dict()
    assert a.keys() == b.keys()
    rows = []
    for key in a:
        delta = (a[key] - b[key]).abs()
        rows.append({'tensor': key, 'max_abs': float(delta.max()), 'mean_abs': float(delta.mean()),
                     'allclose': torch.allclose(a[key], b[key], atol=2e-5, rtol=2e-4)})
    results[name] = rows
    (D / 'fixture_comparisons.json').write_text(json.dumps(results, indent=2))
    assert all(x['allclose'] for x in rows), name
    print('PASS', name, len(rows), flush=True)
