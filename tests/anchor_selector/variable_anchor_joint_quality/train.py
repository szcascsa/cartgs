"""仅训练原 joint 可变成员方案，固定模型复用已有实验。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path('/home/prmi/zwj/variable-anchor-joint-quality-20261021')
repo = Path('/home/prmi/zwj/variable-anchor-20261021/source')
out = root/'validation'
out.mkdir(exist_ok=True)
source = json.loads(Path('/home/prmi/zwj/variable-anchor-20261021/source_manifest.json').read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert all(sha(repo/p) == h for p, h in source.items())
base = Path('/home/prmi/zwj/anchor-slot-deactivation-20261021/enabled_seed2026/input_selector.yaml')
text = base.read_text()+'\nTraining.serial_full_lod: 0\nTraining.serial_lod_detach_master: 0\nTraining.post_topology_full_steps: 0\n'
protocol = dict(scene='fr1_xyz', seed=2026, mapping_steps=10000, training_runs=1,
                methods=['fixed', 'variable'], schedule='original mode1 joint; serial_full_lod=0; Full100=0',
                base_config=str(base), base_config_sha256=sha(base), source_sha256=source,
                baseline='/home/prmi/zwj/anchor-slot-deactivation-20261021/enabled_seed2026',
                only_between_arm_config_change='Selector.variable_anchor_members: 0 -> 1; explicit default-zero serial/Full100 flags',
                slot_deactivation=True, si_gi='disabled', checkpoint='final 10000, no selection by quality',
                frames='reuse prior locked 64 temporal frames when both trajectories cover all; otherwise stop',
                production='Full plus Cached Bernoulli at ratios 0.01,0.05,0.10,0.15',
                same_k='four equally spaced integer budgets in [ceil(0.01*maxN),floor(0.15*minN)]; input ratio K/N per model; cached logit margin exact-K diagnostic only',
                no_tuning=True)
(out/'training_protocol.json').write_text(json.dumps(protocol, indent=2))
env = dict(os.environ, PATH='/usr/local/cuda-11.8/bin:/home/prmi/anaconda3/envs/zwj/bin:'+os.environ['PATH'],
           OMP_NUM_THREADS='2', QT_QPA_PLATFORM='offscreen', MPLBACKEND='Agg', CARTGS_AUDIT_SEED='2026')
runtimes = {}
for name, enabled in [('variable', 1)]:
    run = root/(name+'_seed2026')
    run.mkdir(exist_ok=False)
    config = run/'input_selector.yaml'
    config.write_text(text+'Selector.variable_anchor_members: '+str(enabled)+'\n')
    (out/'status.json').write_text(json.dumps(dict(stage='TRAINING', method=name)))
    cmd = [repo/'bin/tum_rgbd', repo/'third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt',
           repo/'cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_xyz.yaml', repo/'cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml',
           '/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz',
           repo/'cfg/ORB_SLAM3/RGB-D/TUM/associations/fr1_xyz.txt', run, 'no_viewer', '--selector-config', config]
    start = time.monotonic()
    with (run/'training.log').open('w') as log:
        subprocess.run(list(map(str, cmd)), cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    state = json.loads((run/'10000_shutdown/ply/ultrafast_rotation_state.json').read_text())
    assert '10000 iterations completed.' in (run/'training.log').read_text()
    assert not state['serial_full_lod'] and state['post_topology_full_steps'] == 0
    assert state['ultrafast_total_normal_steps'] == 10000
    runtimes[name] = dict(wall_seconds=time.monotonic()-start, strict_benchmark=False)
    (out/'training_runtime.json').write_text(json.dumps(runtimes, indent=2))
    print('TRAINED', name, flush=True)
assert all(sha(repo/p) == h for p, h in source.items())
(out/'status.json').write_text(json.dumps(dict(stage='TRAINED')))
