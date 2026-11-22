"""隔离运行生产SLAM：不覆盖源码、配置、旧实验或默认模型。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/prmi/zwj/anchor-selector-r3-performance-20261022')
SOURCE = Path('/home/prmi/zwj/private-compensation-online-20261022/source')
PREVIOUS = SOURCE.parent
ROOT.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest = json.loads((PREVIOUS/'source_manifest.json').read_text())
assert all(sha(SOURCE/p) == h for p, h in manifest.items())
(ROOT/'source_manifest.json').write_text(json.dumps(manifest, indent=2))
base = (SOURCE/'cfg/selector/selector_config.yaml').read_text()
assert 'Selector.compensation_method: "r3_independent14"' in base
results = []
for name, trace in [('trace1', True), ('untraced', False), ('trace2', True)]:
    run = ROOT/name
    run.mkdir(exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS='2', QT_QPA_PLATFORM='offscreen',
               MPLBACKEND='Agg', CARTGS_AUDIT_SEED='2026',
               LD_LIBRARY_PATH=str(SOURCE/'lib')+':'+os.environ.get('LD_LIBRARY_PATH', ''),
               PATH='/usr/local/cuda-11.8/bin:/home/prmi/anaconda3/envs/zwj/bin:'+os.environ['PATH'])
    for key in ['CARTGS_COMPENSATION_SPLIT4','CARTGS_BUDGET_METHOD','CARTGS_BUDGET_AUDIT',
                'CARTGS_MLP_GRAD_AUDIT','CARTGS_MAPPING_TRACE']:
        env.pop(key, None)
    if trace:
        env['CARTGS_MAPPING_TRACE'] = str(run/'mapping_trace.csv')
    cmd = [SOURCE/'bin/tum_rgbd', SOURCE/'third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt',
           SOURCE/'cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_xyz.yaml',
           SOURCE/'cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml',
           '/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz',
           SOURCE/'cfg/ORB_SLAM3/RGB-D/TUM/associations/fr1_xyz.txt', run,
           'no_viewer', '--selector-config', SOURCE/'cfg/selector/selector_config.yaml']
    start = time.monotonic()
    samples = []
    with (run/'training.log').open('w') as log:
        proc = subprocess.Popen(list(map(str, cmd)), cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            gpu = subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu,clocks.sm,power.draw',
                                           '--format=csv,noheader,nounits'], text=True).strip()
            apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
                                            '--format=csv,noheader,nounits'], text=True).strip()
            samples.append(dict(seconds=time.monotonic()-start, gpu=gpu, apps=apps))
            time.sleep(1)
        assert proc.returncode == 0, (name, proc.returncode)
    results.append(dict(name=name, wall_s=time.monotonic()-start, trace=trace, gpu_samples=samples))
    (ROOT/'online_runs.json').write_text(json.dumps(results, indent=2))
    print('FINISHED', name, results[-1]['wall_s'], flush=True)
assert all(sha(SOURCE/p) == h for p, h in manifest.items())
(ROOT/'online_done.json').write_text(json.dumps(dict(passed=True, source_unchanged=True)))
