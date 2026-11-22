from pathlib import Path
import hashlib
import json
import os
import subprocess
import re

ROOT = Path('/home/prmi/zwj/mycartgs')
D = Path(__file__).resolve().parent
AUDIT = ROOT / 'results/change_distill_optimization/20261017'
expected = json.loads((D / 'expected_normalized_source.json').read_text())
assert all(hashlib.sha256((ROOT / name).read_text().encode()).hexdigest() == digest for name, digest in expected.items())
manifest = json.loads((AUDIT / 'baseline/raw/source_manifest.json').read_text())
unchanged = [name for name in manifest if name not in expected]
assert all((ROOT / name).read_text() == (AUDIT / 'baseline/source' / name).read_text() for name in unchanged)
assert (D / 'build.exit').read_text().strip() == '0'
test_log = (D / 'ctest.log').read_text()
all_tests_passed = bool(re.search(r'100% tests passed, 0 tests failed out of 62', test_log))
known_failure = None
if not all_tests_passed:
    assert (D / 'ctest.exit').read_text().strip() == '8'
    assert '1 tests failed out of 62' in test_log
    failure_list = test_log.split('The following tests FAILED:')[-1]
    assert re.findall(r'\d+ - (\S+) \(', failure_list) == ['ultrafast_mapping_cuda']
    counterfactual = json.loads((D / 'tail_counterfactual.json').read_text())
    assert len(counterfactual) == 4
    assert all(r['exit'] != 0 and any('Mode1 tail advanced clocks without Gaussian/TF optimizer updates' in s
               for s in r['last_lines']) for r in counterfactual)
    known_failure = {'test': 'ultrafast_mapping_cuda', 'also_fails_with_backup_library': True,
                     'also_fails_with_original_binary_and_library': True,
                     'assertion': 'Mode1 tail advanced clocks without Gaussian/TF optimizer updates'}
else:
    assert (D / 'ctest.exit').read_text().strip() == '0'
assert 'Skipped' not in test_log and 'Not Run' not in test_log
old = hashlib.sha256((D / 'backup/lib/libgaussian_mapper.so').read_bytes()).hexdigest()
new = hashlib.sha256((ROOT / 'lib/libgaussian_mapper.so').read_bytes()).hexdigest()
assert new != old
symbols = subprocess.check_output(['nm', '-D', '-C', str(ROOT / 'lib/libgaussian_mapper.so')], text=True)
assert 'selector_cuda::networkMany(' in symbols
env = os.environ.copy()
env['LD_LIBRARY_PATH'] = ':'.join(str(p) for p in [ROOT / 'lib', ROOT / 'third_party/libtorch-cu118/libtorch/lib',
    Path('/usr/local/cuda-11.8/lib64'), ROOT / 'third_party/ORB-SLAM3/lib'])
artifacts = {}
for name in ['lib/libgaussian_mapper.so', 'bin/tum_rgbd', 'bin/tum_mono', 'bin/replica_rgbd',
             'bin/replica_mono', 'bin/vector_mono', 'bin/vector_stereo']:
    p = ROOT / name
    dependencies = subprocess.check_output(['ldd', str(p)], env=env, text=True)
    assert 'not found' not in dependencies, name
    artifacts[name] = {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size,
                       'mtime': p.stat().st_mtime, 'ldd': dependencies}
for name in ['tum_rgbd', 'tum_mono', 'replica_rgbd', 'vector_mono', 'vector_stereo']:
    result = subprocess.run([str(ROOT / 'bin' / name), '--selector-config', str(ROOT / 'cfg/selector/selector_config.yaml')],
                            env=env, capture_output=True, text=True, timeout=30)
    message = result.stdout + result.stderr
    assert 'Usage' in message or 'usage' in message, (name, result.returncode, message)
    (D / (name + '_startup.log')).write_text(message)
data = {'server_project': str(ROOT), 'source_files_updated': list(expected),
        'other_audit_files_unchanged': unchanged, 'build_passed': True, 'ctest_passed': all_tests_passed,
        'tests_total': 62, 'tests_passed': 62 if all_tests_passed else 61, 'existing_test_failure': known_failure,
        'mapper_previous_sha256': old, 'mapper_current_sha256': new, 'artifacts': artifacts,
        'backup': str(D / 'backup'), 'production_source_matches_validated_local_source': True}
(D / 'deployment_manifest.json').write_text(json.dumps(data, indent=2))
print('Production source, shared library, runtime linkage and executable startup PASS;',
      data['tests_passed'], '/62 tests passed; existing failure:', known_failure)
