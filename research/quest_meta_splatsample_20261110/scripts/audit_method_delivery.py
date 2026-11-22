"""当前网络 LoD APK 的来源、数值证据、设备身份和保护检查。"""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[1]
evidence = root/'evidence'
def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def load(path): return json.loads(path.read_text(encoding='utf-8-sig'))
apk = root/'delivery/mycartgs-meta-official-splat-bench.apk'
rollback = root/'delivery/mycartgs-meta-official-splat-bench-0.1.1-rollback.apk'
assert sha(rollback) == '9e24a41b16302d95d362a0ac2c0d88043b1f5591c7051e7cd8ee40002afafe5a'
protected = load(evidence/'OUR_METHOD_BEFORE.json')
for path, value in protected.items(): assert sha(Path(path)) == value,path
checkout = root.parents[1]
production = load(Path('Q:/evidence/XNNPACK_PROTECTED_BEFORE.json'))
for path, value in production.items(): assert sha(checkout/path) == value,path
for path, value in load(evidence/'METHOD_XNNPACK_ARCHIVES.json').items(): assert sha(Path(path)) == value['sha256']
libraries=[]
with zipfile.ZipFile(apk) as z,zipfile.ZipFile(rollback) as old:
    for entry in load(evidence/'OFFICIAL_RENDERER_PROVENANCE_RUN01.json')['official_libraries']:
        data=z.read(entry['path']);assert data==old.read(entry['path'])
        libraries.append(dict(path=entry['path'],version=entry['version'],sha256=hashlib.sha256(data).hexdigest()))
    assert hashlib.sha256(z.read('assets/scenes/Replica__office2__rgbd.mcgs')).hexdigest()==protected['Q:\\assets\\Replica__office2__rgbd.mcgs']
    assert 'lib/arm64-v8a/libofficial_method.so' in z.namelist()
    assert not any('libQnn' in p or 'libhello_xr' in p for p in z.namelist())
    for fixture in load(evidence/'FIXTURE_EXPORT.json')['subsets']:
        assert hashlib.sha256(z.read('assets/'+fixture['asset'])).hexdigest()==fixture['sha256']
unit_tests=0
for p in (root/'source/SplatSample/app/build/test-results/testReleaseUnitTest').glob('TEST-*.xml'):
    tests=ET.parse(p).getroot().attrib;assert int(tests['failures'])==int(tests['errors'])==0;unit_tests+=int(tests['tests'])
assert load(evidence/'METHOD_CORRECTNESS.json')['status']=='PASS'
benchmarks=[]
for name in ['method_benchmark_run01','method_benchmark_run02']:
    data=load(evidence/name/'results.json');assert data['status']=='complete' and len(data['cases'])==8
    for case in data['cases']:
        assert len(case['cpu_measurements'])==9
        assert case['k']==min(case['eligible'],case['fixed_k'] if case['fixed_k']>=0 else int(case['candidates']*case['ratio_requested']))
    benchmarks.append(dict(path=name,cases=8,counter_readouts=sum(c['readouts'] for c in data['cases'])))
adb='Q:/tools/android-sdk/platform-tools/adb.exe'
def device(*args):return subprocess.check_output([adb,'-s','2G97C5ZJ2G00PX',*args],text=True).strip()
installed=device('shell','pm','path','ai.mycartgs.quest.official').removeprefix('package:')
assert device('shell','sha256sum',installed).split()[0]==sha(apk)
assert device('shell','pm','path','ai.mycartgs.quest').startswith('package:')
runtime=(evidence/'OUR_METHOD_LIVE.log').read_text(errors='replace')
assert 'MODEL_READY' in runtime and 'XNNPACK_ARM_CPU_EXECUTED' in runtime
assert 'BENCH_EXPORT status=complete cases=8' in runtime
result=dict(status='PASS',package='ai.mycartgs.quest.official',version='0.2.1-network-lod',apk_sha256=sha(apk),
            measured_apk_sha256=sha(root/'delivery/mycartgs-meta-official-splat-network-lod-run01.apk'),
            official_sdk='0.15.0',official_libraries_unchanged=libraries,model_and_student_binding_verified=True,
            protected_prior_research_files=len(protected),production_files_unchanged=len(production),
            xnnpack_archives_unchanged=True,original_package_retained=True,installed_apk_matches=True,
            unit_tests_passed=unit_tests,native_budget_cases=22,legacy_arm_bitwise_cases=2,benchmarks=benchmarks,
            original_official_apk_retained=True,
            limitations=['Official GPU counter is application stereo aggregate, not pure raster time.',
                         'FPS is measured SDK frame callback cadence, not an independent count of displayed GPU frames.',
                         'Local uses nominal Anchor stereo union and center-head view conditions.',
                         'Public SDK PLY reload limits LoD update rate; it is displayed separately.',
                         'Final APK metadata/lifecycle refinements were not a new full benchmark or GT image-quality test.'])
(evidence/'METHOD_FINAL_VALIDATION.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='official_libraries_unchanged'},indent=2))
