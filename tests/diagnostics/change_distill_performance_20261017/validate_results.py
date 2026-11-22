"""验证插桩未改变训练输出，以及每次重复的输入/step契约。"""
from pathlib import Path
import json
import torch

D = Path(__file__).resolve().parent
RAW = D / 'raw'
results = {}


def parameters(path):
    return torch.jit.load(str(path), map_location='cpu').state_dict()


for branch in ['selector', 'tf']:
    base = RAW / 'cases' / f'{branch}_e5_b8192_r4_m0/warm_0/final_parameters.pt'
    observed = RAW / 'cases' / f'{branch}_e5_b8192_r4_m1/warm_0/final_parameters.pt'
    if not base.exists() or not observed.exists():
        results[branch] = {'status': 'missing'}
        continue
    original, instrumented = parameters(base), parameters(observed)
    assert original.keys() == instrumented.keys()
    fields = []
    for key in original:
        a, b = original[key], instrumented[key]
        fields.append({'parameter': key, 'shape': list(a.shape),
                       'max_abs': float((a-b).abs().max()),
                       'mean_abs': float((a-b).abs().mean()), 'bitwise_equal': bool(torch.equal(a, b)),
                       'allclose': bool(torch.allclose(a, b, atol=2e-5, rtol=2e-4))})
    results[branch] = {'parameter_tensors': len(fields), 'all_bitwise_equal': all(r['bitwise_equal'] for r in fields),
                       'allclose': all(r['allclose'] for r in fields), 'fields': fields}
    assert results[branch]['allclose'], (branch, results[branch])

contracts = []
for path in sorted((RAW / 'cases').glob('*/runs.json')):
    if (path.parent / 'CONTAMINATED.json').exists():
        continue
    for row in json.loads(path.read_text()):
        assert row['identical_initial_parameters'] and row['gaussian_unchanged']
        assert row['normal_optimizer_fresh_and_unchanged']
        branch, batch, epochs = row['branch'], row['batch_size'], row['epochs']
        count = max(row['selector_count'], row['anchor_count']) if branch == 'selector' else row['tf_count']
        expected = (count+batch-1)//batch*epochs
        actual = row['selector_steps'] if branch == 'selector' else row['tf_steps']
        assert actual == expected, (path, row)
        contracts.append({'case': path.parent.name, 'label': row['label'], 'expected_steps': expected,
                           'actual_steps': actual, 'passed': True})
(RAW / 'instrumentation_parity.json').write_text(json.dumps(results, indent=2))
(RAW / 'run_contracts.json').write_text(json.dumps(contracts, indent=2))
print(json.dumps({'cases_checked': len(contracts),
                  'parity': {k: {a: b for a, b in v.items() if a != 'fields'} for k, v in results.items()}}, indent=2))
