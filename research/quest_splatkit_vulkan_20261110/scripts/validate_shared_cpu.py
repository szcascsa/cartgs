from pathlib import Path
import csv
import hashlib
import json
import math
import re
import statistics

root=Path(__file__).resolve().parents[1]
e=root/'evidence/left_eye_shared'
assert 'STEREO_RUNTIME_PASS' in (e/'NATIVE_TEST.log').read_text(errors='replace')
with (e/'NATIVE_CASES.csv').open(encoding='utf-8-sig') as f:stereo=list(csv.DictReader(f))
assert len(stereo)==8
for row in stereo:
    for key in ['left_bitwise','right_shared','render_revision_shared','one_upload']:
        assert row[key]=='1'
    assert row['selector_calls']==row['student_calls']=='1'
    assert float(row['gpu_left_ms'])>0 and float(row['gpu_right_ms'])>0
    assert int(row['left_nonblack'])>0 and int(row['right_nonblack'])>0
parity=json.loads((e/'APP_STEREO_PARITY.json').read_text())
assert parity['status']=='PASS' and parity['frames']>0
sources=sorted((e/'native_cpu_benchmarks').rglob('cpu_inference_*.csv'),key=lambda p:p.name)
assert sources, 'Missing native CPU benchmark CSV'
def summarize_cpu(path,source):
    summaries=[]
    raw=path.read_text(encoding='utf-8-sig')
    assert '# status=complete' in raw
    for key in ['# backend=XNNPACK_ARM_CPU_FP32_2_threads_Selector19_32_32_2_Student62_32_32_14',
                '# teacher_map_uid_fingerprint=', '# student_checkpoint_sha256=']:
        assert key in raw
    samples=list(csv.DictReader(line for line in raw.splitlines() if not line.startswith('#')))
    assert len(samples)==315
    assert 'conditioning_ratio' in samples[0]
    for case in range(9):
        all_rows=[r for r in samples if int(r['case'])==case]
        rows=[r for r in all_rows if r['warmup']=='0']
        assert len(all_rows)==35 and len(rows)==30
        assert len({(r['candidates'],r['eligible'],r['k']) for r in all_rows})==1
        first=rows[0];item=dict(source=source,run=path.stem,case=case,ratio=float(first['ratio']),fixed_k=int(first['fixed_k']),
            candidates=int(first['candidates']),eligible=int(first['eligible']),k=int(first['k']),samples=30)
        if 'conditioning_ratio' in first:item['conditioning_ratio']=float(first['conditioning_ratio'])
        for row in all_rows:
            assert row['selector_calls']==row['student_calls']=='1'
            n=int(row['candidates']);available=int(row['eligible']);fixed=int(row['fixed_k'])
            target=fixed if fixed else math.floor(n*float(row['ratio']))
            assert int(row['k'])==min(available,target)
            for key in ['prepare_ms','selector_ms','topk_ms','student_ms','decode_ms','network_ms','cpu_total_ms']:
                assert math.isfinite(float(row[key])) and float(row[key])>=0
        for key in ['prepare_ms','selector_ms','topk_ms','student_ms','decode_ms','network_ms','cpu_total_ms']:
            values=sorted(float(row[key]) for row in rows)
            item[key+'_median']=statistics.median(values);item[key+'_p95']=values[math.ceil(.95*len(values))-1]
        summaries.append(item)
    assert len(summaries)==9
    return summaries
native_path=sources[-1]
summaries=summarize_cpu(native_path,'Quest3 native device probe')
with (e/'CPU_BENCHMARK_SUMMARY.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
apk_paths=sorted((e/'apk_cpu_benchmarks').glob('cpu_inference_*.csv'))
assert len(apk_paths)>=2, 'Missing completed foreground APK CPU tests'
apk_summaries=[]
for path in apk_paths:
    apk_summaries.extend(summarize_cpu(path,'Quest3 foreground APK 0.2.1'))
    raw=path.read_text()
    for prefix in ['# teacher_map_uid_fingerprint=','# student_checkpoint_sha256=']:
        assert next(x for x in raw.splitlines() if x.startswith(prefix)) in native_path.read_text()
with (e/'CPU_APK_BENCHMARK_SUMMARY.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(apk_summaries[0]));writer.writeheader();writer.writerows(apk_summaries)
baseline=json.loads((root/'evidence/FINAL_VALIDATION.json').read_text())
assert baseline['renderer_library_sha256']=='ed128563d68fe0f5df7b1b09729093f061cd16129b4f54f09dbb26d5da40012c', 'Native library changed after measured CPU tests'
version=re.search(r"versionName '([^']+)'",(root/'host/app/build.gradle').read_text()).group(1)
result=dict(status='PASS',version=version,apk_sha256=baseline['apk_sha256'],
    native_stereo_cases=8,real_gpu_both_eyes=True,left_original_bitwise=True,right_shares_rows_attributes_and_upload=True,
    actual_app_stereo_frames=parity['frames'],user_confirmed_both_eyes_normal=True,
    cpu_sweep_cases=9,cpu_warmup_samples=45,cpu_measured_samples=270,cpu_network_calls_per_sample=dict(selector=1,student=1),
    cpu_scope='fixed current left-eye camera; candidates, input packing, Selector, strict TopK, Student, physical decode; excludes Gaussian render/upload and model loading',
    native_cpu_source=str(native_path.relative_to(root)),native_cpu_source_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
    native_cpu_results=summaries,integrity=baseline,
    foreground_cpu_runs=len(apk_paths),foreground_cpu_measured_samples=270*len(apk_paths),foreground_cpu_results=apk_summaries,
    foreground_cpu_sources=[dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in apk_paths],
    measured_native_library_unchanged=True,
    limitations=['Foreground CPU tests use different fixed left-eye cameras and must not be pooled as same-input repeats.',
        'User visual confirmation applies to the left-shared 0.2.0 implementation; 0.2.1 added CPU benchmark; 0.2.2 only adds explicit CSV export.',
        'Shared Gaussian attributes use the left eye only; the right eye retains its camera but does not add right-only candidates.',
        'No GT image-quality comparison or paired end-to-end FPS speedup is claimed.'])
(e/'VALIDATION.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(dict(status='PASS',native_stereo_cases=8,actual_app_stereo_frames=parity['frames'],cpu_sweep_cases=9,
    native_cpu_measured_samples=270,foreground_cpu_runs=len(apk_paths),foreground_cpu_measured_samples=len(apk_paths)*270),indent=2))
