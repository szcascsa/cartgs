"""原始逐 tensor 误差、固定契约和 native 时序汇总。"""
from pathlib import Path
import json
import statistics
import sys
import torch

D = Path(__file__).resolve().parent
baseline, candidate = [D / name for name in sys.argv[1:3]]
branch = sys.argv[3]
out = candidate / 'raw' / ('comparison_' + baseline.name + '_' + branch + '.json')
result = {'baseline': baseline.name, 'candidate': candidate.name, 'branch': branch}

def compare_files(a, b):
    original = torch.jit.load(str(a), map_location='cpu').state_dict()
    changed = torch.jit.load(str(b), map_location='cpu').state_dict()
    assert original.keys() == changed.keys(), (a, b)
    rows = []
    for name, x in original.items():
        y = changed[name]
        delta = (x - y).abs()
        rows.append({'tensor': name, 'shape': list(x.shape),
                     'max_abs': float(delta.max()) if x.numel() else 0,
                     'mean_abs': float(delta.mean()) if x.numel() else 0,
                     'bitwise_equal': torch.equal(x, y),
                     'allclose': torch.allclose(x, y, atol=2e-5, rtol=2e-4)})
    return {'allclose': all(x['allclose'] for x in rows),
            'bitwise_equal': all(x['bitwise_equal'] for x in rows), 'tensors': rows}

for name, relative in [('verification', Path('verify_' + branch) / 'verification.pt'),
                       ('final_parameters', Path(branch) / 'warm_0/final_parameters.pt')]:
    a, b = baseline / 'raw' / relative, candidate / 'raw' / relative
    if a.exists() and b.exists():
        result[name] = compare_files(a, b)

for version, root in [('baseline', baseline), ('candidate', candidate)]:
    path = root / 'raw' / branch / 'runs.json'
    if not path.exists(): continue
    rows = [r for r in json.loads(path.read_text()) if r['label'].startswith('warm_')]
    for r in rows:
        assert r['gaussian_unchanged'] and r['normal_optimizer_fresh_and_unchanged']
        assert r['identical_initial_parameters'] and r['batch_size'] == 8192
        assert r['ratio_count'] == 4 and r['epochs'] == 5
        assert r['selector_steps'] == (20 if branch in ['selector', 'both'] else 0)
        assert r['tf_steps'] == (20 if branch in ['tf', 'both'] else 0)
    summary = {'runs': len(rows)}
    for field in ['cpu_wall_ms', 'cuda_elapsed_ms', 'additional_allocated_bytes', 'peak_allocated_bytes']:
        values = [r[field] for r in rows]
        summary[field] = {'median': statistics.median(values), 'min': min(values), 'max': max(values),
                          'std': statistics.pstdev(values)}
    result[version + '_timing'] = summary
if 'baseline_timing' in result and 'candidate_timing' in result:
    result['speedup'] = result['baseline_timing']['cpu_wall_ms']['median'] / result['candidate_timing']['cpu_wall_ms']['median']
out.write_text(json.dumps(result, indent=2))
print(json.dumps({k: ({a: b for a, b in v.items() if a != 'tensors'} if isinstance(v, dict) else v)
                  for k, v in result.items()}, indent=2))
assert all(result[k]['allclose'] for k in ['verification', 'final_parameters'] if k in result)
