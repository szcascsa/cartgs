"""交付前核对审计样本、有限值、版本证据与必需文件。"""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

D=Path(__file__).resolve().parent
def data(name):
    return json.loads((D/name).read_text(encoding='utf-8'))
def rows(name):
    return list(csv.DictReader((D/name).open(encoding='utf-8')))

samples=rows('native_samples.csv')
groups=Counter((r['mode'],r['N_total'],r['ratio']) for r in samples)
assert len(samples)==48000 and len(groups)==96 and set(groups.values())=={500}
assert all(math.isfinite(float(r[k])) and float(r[k])>=0 for r in samples for k in ['total_event_ms','wall_ms','K'])
assert all(r['finite']=='1' for r in rows('native_validity.csv'))
isolated=rows('native_isolated_costs.csv')
assert len(isolated)==560
assert len({(r['N'],r['ratio'],r['stage']) for r in isolated})==280
assert all(math.isfinite(float(r[k])) and float(r[k])>=0 for r in isolated for k in ['mean','p50','p95'])
assert sum(p['numel'] for p in data('parameter_shapes.json'))==4962
assert len(data('parameter_shapes.json'))==18
assert len({r['test'] for r in rows('gradient_losses.csv')})==11
assert all(r['finite']=='1' for r in rows('gradient_paths.csv') if r['grad_defined']=='1')
assert all(r['finite']=='1' for r in rows('gradient_losses.csv'))
assert data('gradient_parity_and_graph.json')['all_passed']
assert data('native_parity.json')['staged_hard_exact']
assert data('native_parity.json')['python_probability_max_abs']<2e-6
assert data('native_preview_validity.json')['discard_unchanged']
assert data('native_preview_validity.json')['commit_state_step_1']
ns=data('native_nsys_summary.json')
assert ns['captured_iterations']==10
assert ns['total_iteration_kernel_count']==3510
assert sum(v['kernels_per_iteration'] for v in ns['stages'].values())==351
alloc=data('tensor_allocations.json')
assert sum(v['requests_per_iteration'] for v in alloc['by_operator'])==alloc['allocation_requests_per_iteration']
assert abs(sum(v['requested_bytes_per_iteration'] for v in alloc['by_operator'])-alloc['requested_bytes_per_iteration'])<1
assert data('workload_quantiles.json')['rows']==36228
verification=data('source_verification.json')
assert set(verification['changed_production_sources'])<= {'cfg/selector/fast.yaml'}
before=data('source_hashes_before.json')
assert all(before[k]==v for k,v in verification['source_hashes'].items() if k!='cfg/selector/fast.yaml')
for name in ['run.exit','extra.exit','final.exit']:
    assert (D/name).read_text().strip()=='0'
report=D.parents[2]/'ORIGINAL_SELECTOR_CUDA_AUDIT.md'
assert report.is_file() and '{{' not in report.read_text(encoding='utf-8')
for name in ['native_benchmark.json','native_isolated_costs.json','operator_breakdown.json','tensor_allocations.json',
             'source_verification.json','README.md','build_CMakeCache.txt']:
    assert (D/name).is_file()

manifest={str(p.relative_to(D)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
          for p in D.iterdir() if p.is_file() and p.name not in ['artifact_manifest.json','artifact_validation.json']}
manifest['../../../ORIGINAL_SELECTOR_CUDA_AUDIT.md']=dict(bytes=report.stat().st_size,sha256=hashlib.sha256(report.read_bytes()).hexdigest())
(D/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
result=dict(passed=True,native_samples=len(samples),native_cases=len(groups),isolated_cases=280,gradient_cases=11,
            captured_kernels=3510,parameter_count=4962,observed_workload_rows=36228,
            source_drift=verification['changed_production_sources'],files_hashed=len(manifest))
(D/'artifact_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
