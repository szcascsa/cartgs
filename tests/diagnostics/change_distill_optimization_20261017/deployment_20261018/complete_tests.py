"""构建完成后的正式测试；只将空闲GPU下的计时视为性能证据。"""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys

ROOT = Path('/home/prmi/zwj/mycartgs')
D = Path(__file__).resolve().parent
assert (D / 'build.exit').read_text().strip() == '0'
# The original coordinator may be waiting for an idle GPU. Stop only that
# coordinator after its successful build, never a build/test or external job.
pid = int(sys.argv[1])
process = Path('/proc') / str(pid)
if process.exists():
    assert 'install_and_build.py' in (process / 'cmdline').read_bytes().decode(errors='replace')
    children = (process / 'task' / str(pid) / 'children').read_text().split()
    assert not children, 'Coordinator has an active child; do not interrupt it'
    os.kill(pid, signal.SIGTERM)
env = os.environ.copy()
env['PATH'] = '/usr/local/cuda-11.8/bin:' + env['PATH']
env['LD_LIBRARY_PATH'] = ':'.join(str(p) for p in [ROOT / 'lib', ROOT / 'third_party/libtorch-cu118/libtorch/lib',
    Path('/usr/local/cuda-11.8/lib64'), ROOT / 'third_party/ORB-SLAM3/lib'])
command = ['/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/ctest',
           '--test-dir', str(ROOT / 'build-cuda118-sm86-gcc9'), '--output-on-failure', '-j1']
active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)
(D / 'test_environment.json').write_text(json.dumps({'existing_gpu_processes': active,
    'timing_used_as_performance_evidence': False, 'ctest_command': command}, indent=2))
assert not (D / 'ctest.exit').exists(), 'Tests already completed'
with (D / 'ctest.log').open('w') as log:
    result = subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
(D / 'ctest.exit').write_text(str(result.returncode))
assert result.returncode == 0
subprocess.run(['/home/prmi/anaconda3/envs/zwj/bin/python', str(D / 'verify_deployment.py')], check=True, env=env)
print('Production build and full CTest validation complete', flush=True)
