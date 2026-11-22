"""生成本次最终网络性能表和数学验收摘要；不采用历史审计耗时。"""
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parent


def read(name):
    with (root/name).open(newline='') as stream:
        return list(csv.DictReader(stream))


network = read('round2_network_benchmark.csv')
index = {(r['backend'], int(r['M']), r['stage']): r for r in network if r['metric'] == 'wall'}
lines = ['| M | Forward legacy / strict ms | 倍率 | Backward legacy / strict ms | 倍率 | F+B legacy / strict ms | 倍率 |',
         '|---:|---:|---:|---:|---:|---:|---:|']
for n in (1000, 5000, 10000, 20000, 50000, 100000, 200000, 219969):
    cells = [str(n)]
    for stage in ('network_forward', 'network_backward', 'network_forward_backward'):
        a, b = (float(index[(backend, n, stage)]['mean_ms']) for backend in ('legacy', 'cublas_strict'))
        cells += [f'{a:.4f} / {b:.4f}', f'{a/b:.3f}x']
    lines.append('| '+' | '.join(cells)+' |')
(root/'network_summary_table.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
caller_grid = []
for n in (1000, 5000, 10000, 20000, 50000, 100000, 200000, 219969):
    name = f'grid_caller_{n}.csv'
    if (root/name).exists():
        caller_grid += [dict(M=n, **r) for r in read(name)]
if caller_grid:
    with (root/'caller_grid_benchmark.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(caller_grid[0]))
        writer.writeheader()
        writer.writerows(caller_grid)

parity = {}
names = ['final_strict_219969.csv', 'final_strict_1000.csv', 'strict_random_1000.csv']
names += [p.name for p in sorted(root.glob('grid_strict_*.csv'))]
for name in names:
    if not (root/name).exists():
        continue
    rows = read(name)
    parity[name] = dict(checks=len(rows), failures=sum(r['passed'] != '1' for r in rows),
                        max_abs=max(float(r['max_abs']) for r in rows),
                        max_rms=max(float(r['rms']) for r in rows), max_p99=max(float(r['p99']) for r in rows),
                        unequal_elements=sum(int(r['unequal_elements']) for r in rows))
optimizer = None
if (root/'optimizer_candidate_parity.csv').exists():
    rows = read('optimizer_candidate_parity.csv')
    optimizer = {mode: dict(last_step=max(int(r['step']) for r in rows if r['mode'] == mode),
                            checks=sum(r['mode'] == mode for r in rows),
                            failures=sum(r['exact'] != '1' for r in rows if r['mode'] == mode))
                 for mode in sorted({r['mode'] for r in rows})}
environment = {name: json.loads((root/name).read_text()) for name in ('round2_environment_before.json', 'round2_environment_after.json')}
before, after = environment.values()
verification = {key: before[key] == after[key] for key in ('compiled_sources', 'production_files', 'binaries')}
verification['source_manifest_differences'] = [k for k, v in before['compiled_sources'].items() if v != after['compiled_sources'][k]]
verification['runtime_source_files_stable'] = all(before['compiled_sources'][k] == after['compiled_sources'][k]
    for k in before['compiled_sources'] if not k.endswith('CMakeLists.txt'))
summary = dict(production_accepted=False, parity=parity, optimizer=optimizer, stable_during_network_benchmark=verification,
               real_checkpoint_network_wall=[r for r in network if r['metric'] == 'wall' and r['M'] == '219969'],
               caller_grid_scope='Real audit Gaussian attributes/checkpoint; maturity one, entry zero, cached uniform fixed; Gaussian activation and full Selector input gradients enabled; excludes TF/rasterizer/GI/Gaussian optimizer',
               caller_grid_wall=[r for r in caller_grid if r['metric'] == 'wall'])
summary['initial_network_parity_scope'] = 'Initial audit checkpoint/random fixtures and shape grid; later trained-checkpoint failures are reported separately and block production.'
if (root/'online_checkpoint_parity_summary.json').exists():
    summary['actual_online_checkpoint_parity'] = json.loads((root/'online_checkpoint_parity_summary.json').read_text())
if (root/'online_stage_probe.csv').exists():
    summary['actual_checkpoint_fixed_noise'] = [r for r in read('online_stage_probe.csv') if r['stage']=='fixed_tie_hard']
    summary['actual_checkpoint_fixed_noise_gate_passed'] = all(r['unequal']=='0' for r in summary['actual_checkpoint_fixed_noise'])
if (root/'renderer_gradient_summary.json').exists():
    summary['independent_renderer_gradient'] = json.loads((root/'renderer_gradient_summary.json').read_text())
if (root/'online_quality_comparison.csv').exists():
    rows = read('online_quality_comparison.csv')
    summary['online_quality'] = dict(comparisons=len(rows),cuda_means_outside_observed_legacy_range=sum(
        r['cuda_mean_inside_observed_legacy_range']!='True' for r in rows),
        scope='Three independent seed0 runs per backend and scene; observed range does not establish statistical equivalence.')
(root/'final_evidence_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(initial_parity_files=len(parity), initial_parity_failures=sum(v['failures'] for v in parity.values()),
    actual_checkpoint_failed_cases=summary.get('actual_online_checkpoint_parity',{}).get('failed_cases'),
    actual_fixed_noise_gate_passed=summary.get('actual_checkpoint_fixed_noise_gate_passed'),
    production_accepted=False,stable=verification,optimizer=optimizer),ensure_ascii=False))
