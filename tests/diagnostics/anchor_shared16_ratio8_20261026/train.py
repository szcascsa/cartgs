"""使用常用构建直接进行 fr1_xyz R3/Anchor-ratio14 在线质量对照。"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

root = Path('/home/prmi/zwj/anchor-shared16-ratio8-20261026')
repo = Path('/home/prmi/zwj/mycartgs')
out = root/'validation'
base = (repo/'cfg/selector/selector_config.yaml').read_text()
frames = json.loads(Path('/home/prmi/zwj/budget-shared-audit-20261022/validation/evaluation_plan.json').read_text())['frames']
protocol = dict(scene='fr1_xyz', seeds=[2026, 2027],
    methods=['shared16', 'shared8_ratio8', 'shared16_ratio8', 'r3_independent14'],
    training_ratios=[.01, .05, .1, .15], evaluation_ratios=[.01, .025, .05, .075, .1, .125, .15],
    evaluation_frames=frames,
    interpolation='Anchor methods interpolate latent14; ratio-feature methods also interpolate latent8; raw-ratio weights; all keep log(ratio); R3 output14 interpolation',
    initialization='shared16_ratio8 preserves shared8_ratio8 initial function with extra zero shared features, copied/remapped input columns, and preserved RNG; all MLP methods zero initial compensation; shared16/R3 reference initial Selector differs',
    controls='same source/binary; only Anchor ratio feature layout and R3 compensation method differ; log scalar kept; joint/loss/schedule/Cached Bernoulli/TF/SI/GI unchanged; natural stop',
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
        uses_ratio=method in ('shared8_ratio8','shared16_ratio8')
        actual_method='r3_independent14' if method=='r3_independent14' else 'anchor_ratio_latent14'
        config=re.sub(r'^Selector.compensation_method:.*$', f'Selector.compensation_method: "{actual_method}"', base, flags=re.M)
        config=re.sub(r'^Selector.anchor_ratio_features:.*$', f'Selector.anchor_ratio_features: {int(uses_ratio)}', config, flags=re.M)
        config=re.sub(r'^Selector.anchor_ratio_scalar_input:.*\n?', '', config, flags=re.M)
        config=re.sub(r'^Selector.anchor_ratio_shared_dim:.*\n?', '', config, flags=re.M)
        config+=f'\nSelector.anchor_ratio_scalar_input: "{"log" if uses_ratio else "none"}"\n'
        config+=f'Selector.anchor_ratio_shared_dim: {16 if method=="shared16_ratio8" else 8}\n'
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
