from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time

ROOT = Path('/home/prmi/zwj/mycartgs')
D = Path(__file__).resolve().parent
AUDIT = ROOT / 'results/change_distill_optimization/20261017'
assert (D / 'backup/lib/libgaussian_mapper.so').exists(), 'Production baseline library must be backed up first'
records = json.loads((D / 'source_before.json').read_text())
after = {}
for name, previous in records.items():
    p = ROOT / name
    assert hashlib.sha256(p.read_bytes()).hexdigest() == previous['sha256'], 'Concurrent source edit: ' + name
for name in records:
    p = ROOT / name
    data = (D / 'payload' / name).read_bytes()
    temporary = p.with_name(p.name + '.cd-update.tmp')
    temporary.write_bytes(data)
    shutil.copymode(p, temporary)
    os.replace(temporary, p)
    after[name] = {'sha256': hashlib.sha256(data).hexdigest(),
                   'normalized_sha256': hashlib.sha256(data.decode().replace('\r\n', '\n').encode()).hexdigest()}
(D / 'source_after.json').write_text(json.dumps(after, indent=2))
env = os.environ.copy()
env['PATH'] = '/usr/local/cuda-11.8/bin:' + env['PATH']
env['LD_LIBRARY_PATH'] = ':'.join(str(p) for p in [ROOT / 'lib', ROOT / 'third_party/libtorch-cu118/libtorch/lib',
    Path('/usr/local/cuda-11.8/lib64'), ROOT / 'third_party/ORB-SLAM3/lib'])
cmake = '/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake'
ctest = '/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/ctest'
commands = [[cmake, '--build', str(ROOT / 'build-cuda118-sm86-gcc9'), '-j2'],
            [ctest, '--test-dir', str(ROOT / 'build-cuda118-sm86-gcc9'), '--output-on-failure', '-j1']]
(D / 'production_commands.json').write_text(json.dumps(commands, indent=2))
for command, name in zip(commands, ['build', 'ctest']):
    if name == 'ctest':
        # Other experiments use their own copied runtime. Do not stop them.
        # Keep the old shared libraries as the executable baseline after rebuild.
        with (D / 'gpu_idle_wait.jsonl').open('a') as waiting:
            while True:
                active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                                  '--format=csv,noheader'], text=True).strip()
                if not active: break
                waiting.write(json.dumps({'time': time.time(), 'processes': active}) + '\n')
                waiting.flush()
                time.sleep(15)
        subprocess.run(['/home/prmi/anaconda3/envs/zwj/bin/python', str(D / 'run_benchmark.py'), 'baseline'], check=True, env=env)
    with (D / (name + '.log')).open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    (D / (name + '.exit')).write_text(str(result.returncode))
    assert result.returncode == 0, name + ' failed; inspect saved log'
    print(name, 'PASS', flush=True)
subprocess.run(['/home/prmi/anaconda3/envs/zwj/bin/python', str(D / 'run_benchmark.py'), 'optimized'], check=True, env=env)
shutil.copy2(AUDIT / 'compare.py', D / 'compare.py')
for branch in ['selector', 'tf', 'both']:
    subprocess.run(['/home/prmi/anaconda3/envs/zwj/bin/python', str(D / 'compare.py'),
                    'baseline', 'optimized', branch], check=True, env=env)
print('PRODUCTION BUILD, TESTS AND BENCHMARK COMPLETE', flush=True)
