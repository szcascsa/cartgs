from pathlib import Path
import json

D = Path(__file__).resolve().parent
records = []
for path in D.glob('*/raw/*/runs.json'):
    if path.parent.name.startswith('fixture_'): continue
    for row in json.loads(path.read_text()):
        assert row['batch_size'] == 8192 and row['ratio_count'] == 4 and row['epochs'] == 5
        assert row['identical_initial_parameters'] and row['gaussian_unchanged']
        assert row['normal_optimizer_fresh_and_unchanged'] and not row['allow_tf32_cublas']
        branch = row['branch']
        assert row['selector_steps'] == (20 if branch in ['selector', 'both'] else 0)
        assert row['tf_steps'] == (20 if branch in ['tf', 'both'] else 0)
        if branch in ['selector', 'both']:
            assert row['selector_count'] == 29687 and row['anchor_count'] == 15202
        if branch in ['tf', 'both']:
            assert row['tf_count'] == 29683 and row['skipped_outside'] == 4
        records.append({'case': str(path.relative_to(D)), 'label': row['label'], 'passed': True})
monitors = 0
for path in D.glob('*/raw/*/monitor.json'):
    for row in json.loads(path.read_text()):
        assert all(line.split(',')[0].strip() == str(row['own_pid'])
                   for line in row['gpu_processes'].splitlines()), str(path)
    monitors += 1
for stage in ['stage_a_grouped', 'stage_b', 'stage_c']:
    log = (D / stage / 'tests.log').read_text()
    assert all(x in log for x in ['cpu lifecycle PASS', 'cuda lifecycle PASS', 'cache PASS']), stage
for path in D.glob('*/raw/comparison*.json'):
    obj = json.loads(path.read_text())
    if path.parts[-3] == 'stage_a':
        assert not obj['final_parameters']['allclose']
    else:
        assert all(v['allclose'] for v in obj.values() if isinstance(v, dict) and 'allclose' in v), str(path)
(D / 'validation_summary.json').write_text(json.dumps({'run_contracts': records,
    'monitored_cases': monitors, 'lifecycle_suites': 6, 'cache_suites': 3,
    'expected_failed_bmm_preserved': True, 'all_accepted_correctness_comparisons_pass': True}, indent=2))
print('PASS', len(records), 'run contracts;', monitors, 'monitored cases')
