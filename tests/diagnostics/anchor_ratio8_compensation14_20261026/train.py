"""使用常用构建直接进行 fr1_xyz R3/Anchor-ratio14 在线质量对照。"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

root = Path('/home/prmi/zwj/anchor-ratio8-comp14-20261026')
repo = Path('/home/prmi/zwj/mycartgs')
out = root/'validation'
base = (repo/'cfg/selector/selector_config.yaml').read_text()
frames = json.loads(Path('/home/prmi/zwj/budget-shared-audit-20261022/validation/evaluation_plan.json').read_text())['frames']
protocol = dict(scene='fr1_xyz', seeds=[2026, 2027],
    methods=['anchor16_comp14', 'anchor8_ratio8_comp14', 'r3_independent14'],
    training_ratios=[.01, .05, .1, .15], evaluation_ratios=[.01, .025, .05, .075, .1, .125, .15],
    evaluation_frames=frames,
    interpolation='A interpolates latent14; B interpolates ratio8 and latent14 independently before MLP; R3 interpolates actual14 coefficients; raw ratio linear, endpoint clamp',
    initialization='A Anchor16+logratio input27; B shared8+ratio8 input26, no scalar ratio; both zero compensation and zero latent14; R3 zero direct14',
    controls='same source/binary/config except compensation_method and anchor_ratio_features; original joint, loss, ratio schedule, cached Bernoulli, TF/SI/GI disabled; natural mapping stop',
    limitations='online SLAM topology/poses/update counts can diverge; two seeds on one scene; GPU shared with unrelated process; no unit suite/isolated build',
    source_sha256=json.loads((root/'source_manifest.json').read_text()), tuned_on_eval=False)
(out/'preregistration.json').write_text(json.dumps(protocol, indent=2))
runtime = json.loads((out/'training_runtime.json').read_text()) if (out/'training_runtime.json').exists() else {}
for seed in protocol['seeds']:
    for method in protocol['methods']:
        name = f'{method}_seed{seed}'
        if name in runtime:
            continue
        run = root/name
        run.mkdir(exist_ok=False)
        actual_method = 'r3_independent14' if method=='r3_independent14' else 'anchor_ratio_latent14'
        config = re.sub(r'^Selector.compensation_method:.*$', f'Selector.compensation_method: "{actual_method}"', base, flags=re.M)
        config = re.sub(r'^Selector.anchor_ratio_features:.*$', f'Selector.anchor_ratio_features: {int(method=="anchor8_ratio8_comp14")}', config, flags=re.M)
        (run/'input_selector.yaml').write_text(config)
        env = dict(os.environ, OMP_NUM_THREADS='2', QT_QPA_PLATFORM='offscreen', MPLBACKEND='Agg',
            CARTGS_AUDIT_SEED=str(seed), LD_LIBRARY_PATH=str(repo/'lib')+':'+os.environ.get('LD_LIBRARY_PATH', ''),
            PATH='/usr/local/cuda-11.8/bin:/home/prmi/anaconda3/envs/zwj/bin:'+os.environ['PATH'])
        for key in ['CARTGS_COMPENSATION_SPLIT4', 'CARTGS_BUDGET_METHOD', 'CARTGS_BUDGET_AUDIT', 'CARTGS_MLP_GRAD_AUDIT']:
            env.pop(key, None)
        cmd = [repo/'bin/tum_rgbd', repo/'third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt',
            repo/'cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_xyz.yaml',
            repo/'cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml',
            '/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz',
            repo/'cfg/ORB_SLAM3/RGB-D/TUM/associations/fr1_xyz.txt', run, 'no_viewer',
            '--selector-config', run/'input_selector.yaml']
        (out/'training_status.json').write_text(json.dumps(dict(stage='TRAINING', run=name)))
        start = time.monotonic()
        with (run/'training.log').open('w') as log:
            subprocess.run(list(map(str, cmd)), cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        cp = next(run.glob('*_shutdown/ply'))
        cloud = next((cp/'point_cloud').glob('iteration_*'))
        anchor = json.loads((cloud/'anchor_selector_state.json').read_text())
        runtime[name] = dict(method=method, seed=seed, checkpoint=str(cp), wall_seconds=time.monotonic()-start,
            anchor=anchor, schedule=json.loads((cp/'ultrafast_rotation_state.json').read_text()),
            config_sha256=hashlib.sha256(config.encode()).hexdigest())
        (out/'training_runtime.json').write_text(json.dumps(runtime, indent=2))
        print('TRAINED', name, anchor['iteration'], anchor['gaussian_count'], anchor['anchor_count'], flush=True)
(out/'training_status.json').write_text(json.dumps(dict(stage='TRAINED', completed=len(runtime))))
