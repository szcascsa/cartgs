"""仅新增两次训练：整 Anchor pruning；整 Anchor pruning＋仅下一层满10门槛。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root=Path('/home/prmi/zwj/anchor-only-prune-20261021');repo=root/'source';out=root/'validation'
assert json.loads((out/'correctness.json').read_text())['passed']
baseline=Path('/home/prmi/zwj/shared-selector-appearance-20261021/candidate_seed2026')
base=(baseline/'input_selector.yaml').read_text();assert 'Selector.compensation: 1' in base and 'TransformField.enable: 0' in base
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((root/'source_manifest.json').read_text());assert all(sha(repo/p)==h for p,h in manifest.items())
protocol=dict(scene='fr1_xyz',seed=2026,mapping_steps=10000,training_runs=2,baseline=str(baseline),
    methods=['anchor_only','anchor_only_full10'],checkpoint='final10000, no quality selection',
    schedule='original joint, serial_full_lod=0, Full100=0',
    changes=dict(anchor_only='Gaussian opacity/size quality pruning disabled; original Anchor pruning retained',
                 anchor_only_full10='same as anchor_only, plus ONLY NEXT-LEVEL growing requires source Anchor member_count==10'),
    user_confirmed='same-level growing unrestricted; next-level full-ten gate only',
    unchanged=['Gaussian clone/split including split-parent replacement','alternating topology schedule','new SLAM observation insertion',
               '14D compensation and 27D selector input','TF disabled','photo and ratio loss','Adam/LR/start3000',
               'Cached Bernoulli','original Anchor opacity-window pruning','slot deactivation','SI/GI disabled'],
    source_sha256=manifest,base_input_sha256=sha(baseline/'input_selector.yaml'),
    frames=json.loads(Path('/home/prmi/zwj/shared-selector-appearance-20261021/validation/evaluation_plan.json').read_text())['frames'],
    same_k='four integer K in [ceil(.01*max population),floor(.15*min population)] across all three models; K/N input; cached margin exact-K diagnostic',
    no_tuning=True)
(out/'training_protocol.json').write_text(json.dumps(protocol,indent=2))
env=dict(os.environ,PATH='/usr/local/cuda-11.8/bin:/home/prmi/anaconda3/envs/zwj/bin:'+os.environ['PATH'],
    OMP_NUM_THREADS='2',QT_QPA_PLATFORM='offscreen',MPLBACKEND='Agg',CARTGS_AUDIT_SEED='2026')
runtimes={}
for name,require_full in [('anchor_only',0),('anchor_only_full10',1)]:
    run=root/(name+'_seed2026');run.mkdir(exist_ok=False)
    config=base+'\nSelector.anchor_only_pruning: 1\nSelector.anchor_refine_requires_full: '+str(require_full)+'\n'
    (run/'input_selector.yaml').write_text(config)
    (out/'training_status.json').write_text(json.dumps(dict(stage='TRAINING',method=name)))
    cmd=[repo/'bin/tum_rgbd',repo/'third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt',repo/'cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_xyz.yaml',
        repo/'cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml','/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz',
        repo/'cfg/ORB_SLAM3/RGB-D/TUM/associations/fr1_xyz.txt',run,'no_viewer','--selector-config',run/'input_selector.yaml']
    start=time.monotonic()
    with (run/'training.log').open('w') as log:
        subprocess.run(list(map(str,cmd)),cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    cp=run/'10000_shutdown/ply';state=json.loads((cp/'ultrafast_rotation_state.json').read_text())
    assert state['ultrafast_total_normal_steps']==10000 and not state['serial_full_lod'] and state['post_topology_full_steps']==0
    assert not (cp/'transform_field.pt').exists() and '10000 iterations completed.' in (run/'training.log').read_text()
    anchor=json.loads((cp/'point_cloud/iteration_10000/anchor_selector_state.json').read_text())
    assert anchor['compensation_output_dim']==14 and anchor['anchor_only_pruning']
    assert anchor['anchor_refine_requires_full']==bool(require_full) and anchor['gaussian_pruned_members']==0
    runtimes[name]=dict(wall_seconds=time.monotonic()-start,strict_benchmark=False)
    (out/'training_runtime.json').write_text(json.dumps(runtimes,indent=2))
    print('TRAINED',name,anchor['gaussian_count'],anchor['anchor_count'],flush=True)
assert all(sha(repo/p)==h for p,h in manifest.items())
(out/'training_status.json').write_text(json.dumps(dict(stage='TRAINED')))
