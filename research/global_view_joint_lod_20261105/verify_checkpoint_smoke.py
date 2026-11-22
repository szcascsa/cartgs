"""核验真实 Full 起点的三组短程运行；不渲染、不修改任何模型。"""
import csv
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[3]
groups = ['baseline', 'joint', 'global_control']


def run_folder(name):
    return Path(json.loads((ROOT/'configs'/('manifest_'+name+'.json')).read_text())['output'])


def bitwise(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))


states = {name: torch.jit.load(str(run_folder(name)/'step0_selector.pt'), map_location='cpu').state_dict()
          for name in groups}
reference = states['baseline']
parity = {}
for name, state in states.items():
    assert set(state) == set(reference)
    mismatches = [key for key in state if not bitwise(state[key], reference[key])]
    parity[name] = {'state_tensors': len(state), 'mismatches': mismatches, 'bitwise': not mismatches}
    assert not mismatches, (name, mismatches)
off = torch.jit.load(str(ROOT/'runs/test_off_final/checkpoint/selector.pt'), map_location='cpu').state_dict()
for name in ['zero', 'off_repeat']:
    other = torch.jit.load(str(ROOT/'runs'/('test_'+name+'_final/checkpoint/selector.pt')), map_location='cpu').state_dict()
    mismatches = [key for key in off if not bitwise(off[key], other[key])]
    errors = {key: float((off[key].double()-other[key].double()).abs().max()) for key in mismatches if off[key].numel()}
    rng_equal = {}
    for device in ['cpu', 'cuda']:
        a = torch.jit.load(str(ROOT/'runs/test_off_final'/('global_'+device+'_rng.pt')), map_location='cpu').state_dict()
        b = torch.jit.load(str(ROOT/'runs'/('test_'+name+'_final')/('global_'+device+'_rng.pt')), map_location='cpu').state_dict()
        rng_equal[device] = set(a) == set(b) and all(bitwise(a[key], b[key]) for key in a)
    assert all(rng_equal.values()), (name, rng_equal)
    parity['off_vs_'+name+'_after8'] = {'bitwise': not mismatches, 'mismatch_count': len(mismatches),
        'max_abs_error': max(errors.values(), default=0), 'tensor_max_abs_error': errors, 'same_global_rng': rng_equal,
        'scope': '8-step CUDA fixture; compare unchanged-route repeat before attribution; not universal bitwise guarantee'}
(ROOT/'reports/INITIALIZATION_PARITY.json').write_text(json.dumps(parity, indent=2))
results = []
for name in groups:
    run = run_folder(name)
    verified = json.loads((run/'training_verified.json').read_text())
    state = json.loads((run/'11992_shutdown/ply/real_lod_state.json').read_text())
    rows = list(csv.DictReader((run/'real_lod_steps.csv').open()))
    assert len(rows) == 12 and state['full_training_renders'] == state['master_updates'] == state['selector_updates'] == 12
    trace_file = run/'view_budget_steps.jsonl'
    trace = [json.loads(line) for line in trace_file.read_text().splitlines()] if trace_file.exists() else []
    assert len(trace) == (0 if name == 'baseline' else 12)
    for record in trace:
        assert not record['strict_topk'] and record['gi_weight'] == 0 and record['adam_commits_this_step'] == 1
        assert abs(record['target_K'] - record['candidate_N']*record['ratio']) < 1e-7
        assert abs(record['budget_loss'] - abs(record['K']-record['target_K'])/max(1, record['target_K'])) < 1e-5
    results.append({'group': name, 'verified': verified, 'state': state,
                    'candidate_N_min': min((v['candidate_N'] for v in trace), default=13298),
                    'candidate_N_max': max((v['candidate_N'] for v in trace), default=13298),
                    'local_K_min': min((v['K'] for v in trace), default=None),
                    'local_K_max': max((v['K'] for v in trace), default=None),
                    'mean_mapping_ms': sum(float(v['ms']) for v in rows)/12,
                    'step0_master_sha256': hashlib.sha256((run/'step0_master.ply').read_bytes()).hexdigest()})
assert len({result['step0_master_sha256'] for result in results}) == 1
assert len({(run_folder(name)/'training_keyframes.csv').read_text() for name in groups}) == 1
images = [p for p in (ROOT/'runs').rglob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
assert not images, images[:10]
(ROOT/'reports/CHECKPOINT_SMOKE.json').write_text(json.dumps(
    {'groups': results, 'same_step0_master': True, 'same_keyframes': True, 'zero_images': True}, indent=2))
gradient = {}
for name in ['gradient', 'legacy_real_lod', 'health']:
    gradient[name] = [line for line in (ROOT/'logs'/('test_'+name+'.log')).read_text().splitlines() if line.startswith('PASS')]
(ROOT/'reports/GRADIENT_TEST_EVIDENCE.json').write_text(json.dumps(gradient, indent=2))
print(json.dumps({'initialization': parity, 'gradient': gradient, 'zero_images': True}, indent=2))
