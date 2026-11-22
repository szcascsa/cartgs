"""核验上游、原项目、APK、Shader及测量样本；不修改设备模型或生产。"""
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import zipfile

root=Path(__file__).resolve().parents[1]
evidence=root/'evidence'
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))
upstream=load(evidence/'UPSTREAM_MANIFEST.json')
for name,value in upstream['files'].items():assert sha(root/'source/upstream'/name)==value,name
protected=load(evidence/'PROTECTED_BEFORE.json')
for name,value in protected.items():assert sha(Path(name))==value,name
production=load(Path('Q:/evidence/XNNPACK_PROTECTED_BEFORE.json'))
for name,value in production.items():assert sha(root.parents[1]/name)==value,name
changes=[];shader_files=0
for path in (root/'source/splatkit/packages').rglob('*'):
    if not path.is_file():continue
    relative=path.relative_to(root/'source/splatkit');old=root/'source/upstream'/relative
    if path.suffix in {'.vert','.frag','.comp','.glsl'}:
        assert sha(path)==sha(old);shader_files+=1
    if old.exists() and sha(path)!=sha(old):changes.append(str(relative))
assert len(changes)==3
apk=root/'delivery/mycartgs-splatkit-baseline.apk'
with zipfile.ZipFile(apk) as z:
    assert hashlib.sha256(z.read('assets/scenes/Replica__office2__rgbd.mcgs')).hexdigest()==sha(Path('Q:/assets/Replica__office2__rgbd.mcgs'))
    assert not any(p.startswith('assets/shaders/') for p in z.namelist())
    library_sha=hashlib.sha256(z.read('lib/arm64-v8a/libhello_xr.so')).hexdigest()
adb='Q:/tools/android-sdk/platform-tools/adb.exe'
def device(*args):return subprocess.check_output([adb,'-s','2G97C5ZJ2G00PX',*args],text=True).strip()
installed=device('shell','pm','path','ai.mycartgs.quest.splatkit').removeprefix('package:')
assert device('shell','sha256sum',installed).split()[0]==sha(apk)
retained=['ai.mycartgs.quest','ai.mycartgs.quest.official']
for package in retained:assert device('shell','pm','path',package).startswith('package:')
uids={x['k']:x['uid_sha256'] for x in load(Path('S:/evidence/FIXTURE_EXPORT.json'))['subsets']}
for name in ['pointcount_1200x680_run02.csv','pointcount_1200x680_run03.csv']:
    rows=list(csv.DictReader(l for l in (evidence/name).read_text().splitlines() if not l.startswith('#')))
    assert len(rows)==2070 and sum(v['warmup']=='0' for v in rows)==1800
    for v in rows:
        if int(v['k']):assert v['selected_uid_sha256']==uids[int(v['k'])]
        assert 0<=int(v['drawn'])<=int(v['k'])
for path in (evidence/'apk_benchmarks').glob('*.csv'):
    rows=list(csv.DictReader(l for l in path.read_text().splitlines() if not l.startswith('#')))
    assert len(rows)==324 and sum(v['warmup']=='0' for v in rows)==270
dependencies={name:subprocess.check_output(['git','-C',str(root/'deps'/name),'rev-parse','HEAD'],text=True).strip() for name in ['vk-bootstrap','vma','Vulkan-Headers']}
result=dict(status='PASS',upstream=upstream['commit'],original_upstream_files_verified=len(upstream['files']),
    modified_upstream_code=changes,shader_files_bitwise_unchanged=shader_files,protected_files_unchanged=len(protected),production_files_unchanged=len(production),
    apk_sha256=sha(apk),renderer_library_sha256=library_sha,installed_matches=True,retained_packages=retained,dependencies=dependencies,
    policy=dict(sort_bits=16,subpixel_pixels=1,sh_degree=3,rgba_bits=8,cov_sh='half',hierarchy=False,render_scale=1),
    device_native_tests=['camera_projection','splat_pipeline','radix_sort_144_cases','frame_compute'],native_formal_measured_samples=3600,headset_measured_samples=540,
    limitations=['No GT image-quality evaluation or subjective confirmation yet.','Offscreen timing is one eye; OpenXR presents two sequential eyes.',
        'Native 1200x680 tests and Quest FOV 1680x1760 app tests have different projected loads.','Upstream 12M reference-world and long thermal validation not run.',
        'Radix max-capacity 4M test skipped.','Vulkan validation layer was unavailable in release tests.'])
(evidence/'FINAL_VALIDATION.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
