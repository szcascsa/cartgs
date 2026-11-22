"""在上一轮协议上只扩展补偿输出，原 joint、TF 关闭，最终 step10000 单次训练。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root=Path('/home/prmi/zwj/shared-selector-appearance-20261021');repo=root/'source';out=root/'validation'
assert json.loads((out/'correctness.json').read_text())['passed']
baseline=Path('/home/prmi/zwj/shared-selector-compensation-20261021/candidate_seed2026')
run=root/'candidate_seed2026';run.mkdir(exist_ok=False)
config=(baseline/'input_selector.yaml').read_text()
assert 'TransformField.enable: 0' in config and 'Selector.compensation: 1' in config
(run/'input_selector.yaml').write_text(config)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((root/'source_manifest.json').read_text());assert all(sha(repo/p)==h for p,h in manifest.items())
protocol=dict(scene='fr1_xyz',seed=2026,mapping_steps=10000,training_runs=1,baseline=str(baseline),
    checkpoint='final 10000, no quality selection',schedule='original joint, serial_full_lod=0, Full100=0',
    changes=['10D geometry head -> 14D geometry + opacity-logit + SH DC head, zero initialization',
             'CD preserves newly compensated opacity and SH DC as well as geometry'],
    unchanged=['exact same input YAML','27D input/shared encoder/selection branch','TF disabled','original reconstruction and ratio loss',
               'Selector Adam/LR','start iteration 3000','Cached Bernoulli','variable Anchor lifecycle','SI/GI disabled'],
    input_config_sha256=sha(run/'input_selector.yaml'),source_sha256=manifest,
    frames=json.loads(Path('/home/prmi/zwj/shared-selector-compensation-20261021/validation/evaluation_plan.json').read_text())['frames'],
    same_k='four integer K in [ceil(.01*maxN),floor(.15*minN)]; per-model input K/N; cached-margin exact-K diagnostic only',
    diagnostic='same candidate checkpoint and selected rows; appearance OFF retains geometry compensation',no_tuning=True)
(out/'training_protocol.json').write_text(json.dumps(protocol,indent=2))
env=dict(os.environ,PATH='/usr/local/cuda-11.8/bin:/home/prmi/anaconda3/envs/zwj/bin:'+os.environ['PATH'],
         OMP_NUM_THREADS='2',QT_QPA_PLATFORM='offscreen',MPLBACKEND='Agg',CARTGS_AUDIT_SEED='2026')
cmd=[repo/'bin/tum_rgbd',repo/'third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt',repo/'cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_xyz.yaml',
    repo/'cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml','/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz',
    repo/'cfg/ORB_SLAM3/RGB-D/TUM/associations/fr1_xyz.txt',run,'no_viewer','--selector-config',run/'input_selector.yaml']
(out/'training_status.json').write_text(json.dumps(dict(stage='TRAINING')));start=time.monotonic()
with (run/'training.log').open('w') as log:
    subprocess.run(list(map(str,cmd)),cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
cp=run/'10000_shutdown/ply';state=json.loads((cp/'ultrafast_rotation_state.json').read_text())
assert state['ultrafast_total_normal_steps']==10000 and not state['serial_full_lod'] and state['post_topology_full_steps']==0
assert not (cp/'transform_field.pt').exists() and '10000 iterations completed.' in (run/'training.log').read_text()
anchor=json.loads((cp/'point_cloud/iteration_10000/anchor_selector_state.json').read_text())
assert anchor['selector_compensation'] and anchor['compensation_forwards']>0 and anchor['compensation_output_dim']==14
assert all(sha(repo/p)==h for p,h in manifest.items())
(out/'training_runtime.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-start,strict_benchmark=False),indent=2))
(out/'training_status.json').write_text(json.dumps(dict(stage='TRAINED')));print('TRAINED',flush=True)
