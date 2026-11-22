"""验证交付清单、证据SHA、生产源码未变及关键测量不变量。"""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

D=Path(__file__).resolve().parent;R=D.parents[2];E=D/'evidence'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((E/'evidence_manifest.json').read_text())
assert all(sha(E/p)==x['sha256'] for p,x in manifest.items())
online_manifest=json.loads((D/'evidence_online/online_profile_manifest.json').read_text())
assert all(sha(D/'evidence_online'/p)==x['sha256'] for p,x in online_manifest.items())
whole=json.loads((D/'evidence_online/online_profile_summary.json').read_text())
assert whole['kernel_launches']==1327411 and whole['anchor_state']['compensation_forwards']==401
source=json.loads((E/'source_manifest.json').read_text())
mismatch=[p for p,h in source.items() if not (R/p).is_file() or sha(R/p)!=h]
assert not mismatch,mismatch[:10]
def git(*args):return subprocess.check_output(['git',*args],cwd=R,text=True).strip()
assert git('rev-parse','HEAD')=='4269d7e3bad6cf5af7e1df1763634b61e98e0b51'
assert not git('diff','--name-only') and not git('diff','--cached','--name-only')
required=['ANCHOR_SELECTOR_R3_PERFORMANCE_AUDIT.md','performance_hotspots.csv','kernel_launch_breakdown.csv',
          'cpu_gpu_timeline.csv','optimization_opportunities.csv','cuda_kernel_candidates.md','parity_benchmark_plan.md']
assert all((R/p).stat().st_size>0 for p in required)
assert '@@' not in (R/required[0]).read_text(encoding='utf-8')
def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
launch=read(R/'kernel_launch_breakdown.csv')
assert sum(float(x['launches_per_iteration']) for x in launch if x['case']=='lod_0.050000')==418
assert sum(float(x['launches_per_iteration']) for x in launch if x['case']=='lod_0.050000' and x['stage']=='adam')==112
native=read(E/'native1/native_stages.csv')
assert len({r['case'] for r in native})==5
assert {int(r['K']) for r in native}=={15891,156,799,1582,2368}
f=json.loads((E/'fixture_verification.json').read_text())
assert f['passed'] and len(f['comparisons'])==789 and len(f['fixed_state_adam_checks'])==62
assert not f['end_to_end_allclose'], 'Observed nondeterministic exception must remain visible'
assert (E/'anchor_input_buffers.pt').is_file()
assert 'PASS private R2/R3' in (E/'lifecycle_private.log').read_text(encoding='utf-8')
assert 'PASS variable anchors' in (E/'lifecycle_mapping.log').read_text(encoding='utf-8')
extra=['anchor_input_buffers.pt','fixture_details.json','lifecycle_private.log','lifecycle_mapping.log']
result=dict(passed=True,head=git('rev-parse','HEAD'),tracked_and_index_unchanged=True,
            source_files_verified=len(source),evidence_files_verified=len(manifest),online_evidence_files_verified=len(online_manifest),
            required_outputs={p:dict(bytes=(R/p).stat().st_size,sha256=sha(R/p)) for p in required},
            additional_evidence={p:dict(bytes=(E/p).stat().st_size,sha256=sha(E/p)) for p in extra},
            fixture_comparisons=789,fixed_state_adam_checks=62,end_to_end_allclose=False,
            native_master_gradient_groups=6,production_process_kernels=1327411,production_process_timeline_rows=1378798,
            independent_gpu_benchmark=False)
(D/'delivery_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['required_outputs','additional_evidence']},ensure_ascii=False))
