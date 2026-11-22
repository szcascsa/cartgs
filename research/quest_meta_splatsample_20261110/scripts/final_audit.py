"""核验交付来源、测试数据与受保护文件；仅读取模型、APK 和设备。"""
from pathlib import Path
import csv
import hashlib
import io
import json
import statistics
import subprocess
import xml.etree.ElementTree as ET
import zipfile

root = Path(__file__).resolve().parents[1]
checkout = root.parents[1]
evidence = root / 'evidence'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


apk = root / 'delivery/mycartgs-meta-official-splat-bench.apk'
initial = root / 'delivery/mycartgs-meta-official-splat-bench-run01.apk'
assert sha(apk) == '9e24a41b16302d95d362a0ac2c0d88043b1f5591c7051e7cd8ee40002afafe5a'
assert sha(initial) == 'e37f3883937f47876876f8e47842cf2893727b3b2440bbf9b51debfd2c627d2d'
provenance = load(evidence / 'OFFICIAL_RENDERER_PROVENANCE_RUN01.json')
libraries = []
with zipfile.ZipFile(apk) as z, zipfile.ZipFile(initial) as old:
    for lib in provenance['official_libraries']:
        module, version, name = lib['module'], lib['version'], lib['path']
        candidates = list((root / '.gradle-user/caches/modules-2/files-2.1/com.meta.spatial' / module / version).rglob('*.aar'))
        assert len(candidates) == 1, (module, candidates)
        with zipfile.ZipFile(candidates[0]) as aar:
            data = z.read(name)
            assert data == aar.read(name.replace('lib/', 'jni/', 1)) == old.read(name)
            libraries.append(dict(module=module, version=version, path=name,
                                  sha256=hashlib.sha256(data).hexdigest(), matches_official_aar=True))
    manifest = load(evidence / 'FIXTURE_EXPORT.json')
    for item in manifest['subsets']:
        name = 'assets/' + item['asset']
        assert hashlib.sha256(z.read(name)).hexdigest() == item['sha256']
        assert z.read(name) == old.read(name)
    assert not any('hello_xr' in name or name.endswith('raster.comp.spv') for name in z.namelist())
protected = load(Path('Q:/evidence/XNNPACK_PROTECTED_BEFORE.json'))
for name, expected in protected.items():
    assert sha(checkout / name) == expected, name
assert sha(Path('Q:/assets/Replica__office2__rgbd.mcgs')) == manifest['source_sha256']
assert sha(Path('Q:/delivery/mycartgs-quest-xnnpack-cpu-release.apk')) == 'a8115481b9427338b3dffccfc9c0cfcf7197be4e6c9b4bb4bfb78792bf7a71a5'
upstream = load(evidence / 'UPSTREAM_MANIFEST.json')
for name, expected in upstream['files'].items():
    assert sha(root / 'source/upstream' / name) == expected, name
raw = (evidence / 'official_run_01.csv').read_text(encoding='utf-8-sig')
labeled = (evidence / 'official_run_01_labeled.csv').read_text(encoding='utf-8-sig')
data_lines = lambda text: '\n'.join(line for line in text.splitlines() if not line.startswith('#'))
assert data_lines(raw) == data_lines(labeled)
rows = list(csv.DictReader(io.StringIO(data_lines(raw))))
assert len(rows) == 3902 == len({r['frame_id'] for r in rows})
summary = load(evidence / 'RESULTS.json')
assert len({r['app_gpu_ms'] for r in rows}) == summary['global_distinct_gpu_values'] == 72
for item in summary['results']:
    subset = [r for r in rows if int(r['k']) == item['k']]
    assert len(subset) == item['polls']
    assert {int(r['round']) for r in subset} == {1, 2}
    assert statistics.median(float(r['app_gpu_ms']) for r in subset) == item['app_gpu_median_ms']
tests = ET.parse(root / 'source/SplatSample/app/build/test-results/testReleaseUnitTest/TEST-com.meta.spatial.samples.splatsample.BenchmarkDataTest.xml').getroot()
assert tests.attrib['tests'] == '3' and tests.attrib['failures'] == tests.attrib['errors'] == '0'
adb = 'Q:/tools/android-sdk/platform-tools/adb.exe'
def device(*args):
    return subprocess.check_output([adb, '-s', '2G97C5ZJ2G00PX', *args], text=True).strip()
device_apk = device('shell', 'pm', 'path', 'ai.mycartgs.quest.official').removeprefix('package:')
assert device('shell', 'sha256sum', device_apk).split()[0] == sha(apk)
assert device('shell', 'pm', 'path', 'ai.mycartgs.quest').startswith('package:')
result = dict(status='PASS', package='ai.mycartgs.quest.official', version='0.1.1-meta-official',
              apk_sha256=sha(apk), measured_apk_sha256=sha(initial), actual_official_sdk='0.15.0',
              official_libraries=libraries, fixtures_unchanged=True, source_mcgs_unchanged=True,
              original_apk_unchanged=True, protected_files_verified=len(protected),
              upstream_files_verified=len(upstream['files']), raw_rows=len(rows),
              gpu_distinct_values=72, corrected_csv_changes_data=False, unit_tests_passed=3,
              installed_apk_matches=True, original_package_retained=True,
              limits=['system aggregated readouts, not independent GPU-frame timings',
                      'global static UID-hash subsets; no Selector or Compensator',
                      'current APK label-only update not re-benchmarked'])
(evidence / 'FINAL_INTEGRITY.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'official_libraries'}, indent=2))
