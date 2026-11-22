"""功能门控通过后，一次真实 SLAM/Mapping；不据此声明质量优于 baseline。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path('/home/prmi/zwj/variable-anchor-20261021')
repo = root/'source'
out = root/'validation'
assert json.loads((out/'status.json').read_text())['stage'] == 'CORRECTNESS_PASSED'
run = root/'variable_seed2026'
run.mkdir(exist_ok=False)
base = Path('/home/prmi/zwj/serial-joint-lod-20261021/joint_seed2026/input_selector.yaml')
config = run/'input_selector.yaml'
config.write_text(base.read_text()+'\nSelector.variable_anchor_members: 1\n')
protocol = dict(scene='fr1_xyz', seed=2026, mapping_steps=10000, training_runs=1,
                base_config=str(base), base_config_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
                config_change='Selector.variable_anchor_members: 1 only',
                schedule='unchanged serial joint: Full Master, then live-Master LoD joint',
                topology='alternate eligible Full topology events: Gaussian then Anchor',
                capacity=10, initialization='real distinct observations only, deterministic FPS if more than ten',
                thresholds='existing Gaussian thresholds and Octree-GS Anchor opacity rule unchanged',
                quality_claim=False)
(out/'smoke_protocol.json').write_text(json.dumps(protocol, indent=2))
env = dict(os.environ, PATH='/usr/local/cuda-11.8/bin:/home/prmi/anaconda3/envs/zwj/bin:'+os.environ['PATH'],
           OMP_NUM_THREADS='2', QT_QPA_PLATFORM='offscreen', MPLBACKEND='Agg', CARTGS_AUDIT_SEED='2026')
args = [repo/'bin/tum_rgbd', repo/'third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt',
        repo/'cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_xyz.yaml', repo/'cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml',
        '/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz',
        repo/'cfg/ORB_SLAM3/RGB-D/TUM/associations/fr1_xyz.txt', run, 'no_viewer', '--selector-config', config]
start = time.monotonic()
with (run/'training.log').open('w') as log:
    result = subprocess.run(list(map(str, args)), cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
(out/'smoke_result.json').write_text(json.dumps(dict(exit_code=result.returncode,
    wall_seconds=time.monotonic()-start, strict_benchmark=False), indent=2))
assert result.returncode == 0
assert '10000 iterations completed.' in (run/'training.log').read_text()
print('PASS one real 10000-Mapping smoke; quality improvement not established', flush=True)
