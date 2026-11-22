"""以相同 Full 起点串行执行三组 12 步验收，不启动完整质量实验。"""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
env = dict(os.environ)
env['LD_LIBRARY_PATH'] = ':'.join([
    str(ROOT/'runtime/lib'),
    '/home/prmi/zwj/experiments/real_lod_full_joint_20261103/source/third_party/ORB-SLAM3/lib',
    '/home/prmi/zwj/mycartgs/third_party/install/opencv-cuda118-sm86-gcc9/lib',
    '/home/prmi/zwj/mycartgs/third_party/install/jsoncpp-1.9.5/lib',
    '/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/lib',
    '/usr/local/cuda-11.8/lib64',
])
env['OMP_NUM_THREADS'] = '2'
results = []
for name in ['baseline', 'joint', 'global_control']:
    manifest = ROOT/'configs'/('manifest_'+name+'.json')
    value = json.loads(manifest.read_text())
    output = ROOT/'runs'/('smoke_'+name+'_final')
    assert not output.exists(), f'Preserve prior run: {output}'
    value['output'] = str(output)
    manifest.write_text(json.dumps(value, indent=2))
    started = time.monotonic()
    command = [str(ROOT/'runtime/bin/lod_from_full'), str(manifest), '12']
    with (ROOT/'logs'/('smoke_'+name+'_final.log')).open('w') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        maps = Path(f'/proc/{child.pid}/maps')
        while child.poll() is None:
            try:
                mapping = maps.read_text()
                if str(ROOT/'runtime/lib/libgaussian_mapper.so') in mapping:
                    (ROOT/'reports'/('runtime_maps_'+name+'.txt')).write_text(mapping)
                    break
            except FileNotFoundError:
                pass
            time.sleep(.01)
        returncode = child.wait()
    results.append({'group': name, 'command': command, 'returncode': returncode, 'seconds': time.monotonic()-started})
    (ROOT/'reports/CHECKPOINT_SMOKE_EXECUTION.json').write_text(json.dumps(results, indent=2))
    print(name, returncode, flush=True)
    if returncode:
        raise SystemExit(f'{name} failed; inspect failure before retry')
subprocess.run(['/home/prmi/anaconda3/envs/zwj/bin/python', str(ROOT/'source/research/global_view_joint_lod_20261105/verify_checkpoint_smoke.py')], check=True)
