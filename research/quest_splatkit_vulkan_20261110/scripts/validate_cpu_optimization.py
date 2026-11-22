"""验证最终CPU版本、原始样本、逐位合同和已披露的渲染重复差异。"""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics
import zipfile

root=Path(__file__).resolve().parents[1]
e=root/'evidence/cpu_optimization'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def readcsv(path):
    return list(csv.DictReader(l for l in path.read_text(encoding='utf-8-sig').splitlines() if not l.startswith('#')))
frozen={Path(k).as_posix():v for k,v in json.loads((e/'FROZEN_SOURCE_SHA256.json').read_text()).items()}
changed=[name for name,digest in frozen.items() if sha(root/name)!=digest]
allowed=['host/app/src/main/cpp/mycartgs/splatkit_gaussians.cpp','host/app/src/main/cpp/mycartgs/splatkit_gaussians.h','tests/stereo_runtime_probe.cpp']
assert set(changed)==set(allowed), changed
baseline={Path(k).as_posix():v for k,v in json.loads((e/'BASELINE_MANIFEST.json').read_text()).items()}
for name in ['decode.cpp','residual48.cpp']:
    relative='host/app/src/main/cpp/mycartgs/'+name
    assert sha(root/relative)==baseline[relative]
with zipfile.ZipFile(e/'baseline/source_0.2.2.zip') as z:
    assert "versionName '0.2.2-cpu-csv-export'" in z.read('host/app/build.gradle').decode()
assert sha(e/'baseline/mycartgs-splatkit-baseline.apk')==baseline['delivery/mycartgs-splatkit-baseline.apk']
for log in ['CORRECTNESS_LARGE_BATCH.log','ASAN_UBSAN_NO_VPTR.log']:
    text=(e/log).read_text(errors='replace')
    assert 'CPU_CHECK_PASS bitwise_frames=2305 network_shapes=17 flags=12' in text
    assert 'runtime error:' not in text and 'ERROR: AddressSanitizer' not in text
assert 'runtime error:' in (e/'ASAN_STDLIB_REPRO.log').read_text(errors='replace')
assert 'STEREO_RUNTIME_PASS cases=8' in (e/'STEREO_RGBA.log').read_text(errors='replace')
stereo=readcsv(e/'STEREO_CASES.csv');assert len(stereo)==8
for row in stereo:
    for key in ['left_bitwise','right_shared','render_revision_shared','one_upload']:
        assert row[key]=='1'
    assert row['selector_calls']==row['student_calls']=='1'
    assert int(row['left_nonblack'])>0 and int(row['right_nonblack'])>0
    if row['rgba_left_bitwise']=='0' or row['rgba_right_bitwise']=='0':
        assert row['rgba_no_upload_repeat_bitwise']=='0' and int(row['control_changed_bytes'])>0
native=[]
for name in ['cpu_opt_final_bench_motion.csv','cpu_opt_final_bench_holdout_motion.csv']:
    rows=readcsv(e/name);assert len(rows)==7776
    assert {r['threads'] for r in rows if r['variant']=='original_022'}=={'2'}
    assert {r['threads'] for r in rows if r['variant']=='optimized'}=={'4'}
    assert {r['flags'] for r in rows if r['variant']=='optimized'}=={'221'}
    native.extend(rows)
paired=readcsv(e/'PAIRED_RESULTS.csv')
final=[r for r in paired if r['source'].startswith('cpu_opt_final_')]
assert len(final)==216 and all(float(r['speedup'])>1 for r in final)
frontend=[];sources=[]
for path in sorted((e/'apk_cpu_benchmarks').glob('*.csv')):
    text=path.read_text();meta=dict(l[2:].split('=',1) for l in text.splitlines() if l.startswith('# ') and '=' in l)
    if meta.get('cpu_optimization_flags')!='221':continue
    assert meta['status']=='complete' and meta['backend']=='XNNPACK_ARM_CPU_FP32_4_threads_Selector19_32_32_2_Student62_32_32_14'
    assert meta['student_checkpoint_sha256']=='fd85285c6bd5871f4b41c85690597ccadc965d5a5ec6f474a983e4e09f547a75'
    assert meta['scene']=='Replica__office2__rgbd' and meta['teacher_map_uid_fingerprint']=='9e2d413fa7864127465394fa22bb3da4e05d912d4c9e0822a4e928933aa36804'
    rows=readcsv(path);assert len(rows)==315
    for case in range(9):
        all_rows=[r for r in rows if int(r['case'])==case];measured=[r for r in all_rows if r['warmup']=='0']
        assert len(all_rows)==35 and len(measured)==30
        assert len({(r['candidates'],r['eligible'],r['k']) for r in all_rows})==1
        first=measured[0]
        item=dict(run=path.stem,case=case,ratio=float(first['ratio']),fixed_k=int(first['fixed_k']),
            N=int(first['candidates']),eligible=int(first['eligible']),K=int(first['k']),samples=30)
        for r in all_rows:
            assert r['selector_calls']==r['student_calls']=='1'
            requested=int(r['fixed_k']) or math.floor(int(r['candidates'])*float(r['ratio']))
            assert int(r['k'])==min(requested,int(r['eligible']))
        for key in ['prepare_ms','selector_ms','topk_ms','student_ms','decode_ms','network_ms','cpu_total_ms']:
            values=sorted(float(r[key]) for r in measured)
            assert all(math.isfinite(v) and v>=0 for v in values)
            item[key+'_median']=statistics.median(values);item[key+'_p95']=values[28]
        frontend.append(item)
    sources.append(dict(path=str(path.relative_to(root)),sha256=sha(path),metadata=meta))
assert len(sources)==2 and len(frontend)==18
with (e/'APK_CPU_SUMMARY.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(frontend[0]));writer.writeheader();writer.writerows(frontend)
integrity=json.loads((root/'evidence/FINAL_VALIDATION.json').read_text())
assert integrity['installed_matches'] and integrity['apk_sha256']==sha(root/'delivery/mycartgs-splatkit-baseline.apk')
result=dict(status='PASS_CPU_BITWISE_AND_DEVICE_TESTS',version='0.3.0-cpu-fast',flags=221,threads=4,batch=16384,
    scene='Replica__office2__rgbd',numeric_frames=2305,network_shapes=17,numeric_flags=12,
    asan=True,ubsan=True,excluded_sanitizer_checks=['vptr: reproduced in std::ifstream-only program','LeakSanitizer disabled; not a leak check'],
    native_measured_samples=len(native),native_paired_groups=len(final),native_slower_groups=sum(float(r['speedup'])<1 for r in final),
    foreground_measured_samples=540,foreground_runs=sources,foreground_summary=frontend,
    native_cpu_source_unchanged_after_freeze=True,post_freeze_diagnostic_only_changes=changed,
    stereo_cases=8,stereo_inputs_bitwise=True,one_inference_and_one_upload_per_stereo_frame=True,
    rgba_bitwise_to_original=all(r['rgba_left_bitwise']==r['rgba_right_bitwise']=='1' for r in stereo),
    original_renderer_repeat_nondeterminism_cases=sum(int(r['control_changed_bytes'])>0 for r in stereo),
    limitations=['One Quest 3 and one scene; Quest 3S and other scenes not measured.',
        'Native paired CPU measurements exclude OpenXR/present and cannot stand in for app FPS.',
        'Two new foreground runs differ in pose and runtime load; not same-input repeats.',
        'Original renderer gives different RGBA on exact same repeated inputs. No CPU numeric differences are accepted on that basis.',
        'No new GT PSNR, SSIM, LPIPS or long thermal-session evaluation.'],integrity=integrity)
(e/'VALIDATION.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:result[k] for k in ['status','numeric_frames','native_measured_samples','native_paired_groups','native_slower_groups','foreground_measured_samples','stereo_cases','rgba_bitwise_to_original','original_renderer_repeat_nondeterminism_cases']},indent=2))
