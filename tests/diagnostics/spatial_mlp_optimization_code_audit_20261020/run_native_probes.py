"""保存模型上的12个Bank backward探针；无训练、无权重保存、无eval读取。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT.parent / 'spatial-mlp-implicit-si-20261020'
cmake = '/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake'
env = dict(os.environ, CC='/usr/bin/gcc-9', CXX='/usr/bin/g++-9',
           CUDACXX='/usr/local/cuda-11.8/bin/nvcc', CUDA_HOME='/usr/local/cuda-11.8')
for name, cmd in [
    ('configure', [cmake, '-S', str(ROOT), '-B', str(ROOT / 'build'),
                   '-DCMAKE_CXX_COMPILER=/usr/bin/g++-9',
                   '-DTorch_DIR=/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/share/cmake/Torch']),
    ('compile', [cmake, '--build', str(ROOT / 'build'), '-j1']),
]:
    with (ROOT / (name + '.log')).open('w') as out:
        subprocess.run(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, check=True)

results = []
for scene in ['fr1_room', 'fr1_xyz', 'fr2_xyz']:
    for step in [0, 250, 1000, 2000]:
        weights = LEGACY / scene / 'models' / 'MLP_5_seed_2026' / f'step_{step}_weights.pt'
        sha_before = hashlib.sha256(weights.read_bytes()).hexdigest()
        out_path = ROOT / f'probe_{scene}_{step}.json'
        cmd = [str(ROOT / 'build/gradient_probe'), str(LEGACY / scene / 'training_manifest.json'),
               str(LEGACY / scene / 'training_input.bin'), str(weights), '.05', '2026', str(out_path)]
        runtime = dict(env, LD_LIBRARY_PATH=str(LEGACY / scene / 'runtime/lib') + ':' + env.get('LD_LIBRARY_PATH', ''))
        with (ROOT / f'probe_{scene}_{step}.log').open('w') as out:
            subprocess.run(cmd, env=runtime, stdout=out, stderr=subprocess.STDOUT, check=True)
        result = json.loads(out_path.read_text())
        assert hashlib.sha256(weights.read_bytes()).hexdigest() == sha_before
        assert result['frozen_exact'] and result['optimizer_steps'] == 0
        assert result['shift100_sample_exact']
        assert result['sigmoid_chain_rule_max_error'] < 1e-10
        result.update(scene=scene, step=step, checkpoint_sha_unchanged=sha_before)
        results.append(result)
        print(scene, step, result['sum_output_grad'], result['common_weight_cosine'], flush=True)
(ROOT / 'native_gradient_probes.json').write_text(json.dumps(dict(
    all_passed=True, probes=results, scope='3 scenes x 4 saved checkpoints, ratio5 seed2026, first Bank frame only; no optimizer'), indent=2) + '\n')
