"""编译修复版，执行CPU/CUDA回归、真实Bank backward和3x100步有限sanity。

不读取non-Bank质量，不调LR，不启动正式训练矩阵；历史文件只读。
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parent
LEGACY = ROOT.parent / 'spatial-mlp-implicit-si-20261020'
cmake = '/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake'
env = dict(os.environ, CC='/usr/bin/gcc-9', CXX='/usr/bin/g++-9',
           CUDACXX='/usr/local/cuda-11.8/bin/nvcc', CUDA_HOME='/usr/local/cuda-11.8')
for name, cmd in [
    ('configure', [cmake, '-S', str(ROOT), '-B', str(ROOT / 'build'),
                   '-DCMAKE_CXX_COMPILER=/usr/bin/g++-9',
                   '-DFIXED_WORKER=' + str(ROOT / 'source/train_worker.cpp'),
                   '-DTorch_DIR=/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/share/cmake/Torch']),
    ('compile', [cmake, '--build', str(ROOT / 'build'), '-j1']),
]:
    with (ROOT / (name + '.log')).open('w') as out:
        subprocess.run(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, check=True)


def run(executable, args, log, scene='fr1_room'):
    runtime = dict(env, LD_LIBRARY_PATH=str(LEGACY / scene / 'runtime/lib') + ':' + env.get('LD_LIBRARY_PATH', ''))
    with log.open('w') as out:
        subprocess.run([str(ROOT / 'build' / executable), *map(str, args)],
                       env=runtime, stdout=out, stderr=subprocess.STDOUT, check=True)


run('selection_regression', [], ROOT / 'selection_regression.json')
regression = json.loads((ROOT / 'selection_regression.json').read_text())
results, probes = [], []
for scene in ['fr1_room', 'fr1_xyz', 'fr2_xyz']:
    source = LEGACY / scene
    weights = source / 'models/MLP_5_seed_2026/step_2000_weights.pt'
    old_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    probe_path = ROOT / f'{scene}_probe.json'
    run('native_probe', [source / 'training_manifest.json', source / 'training_input.bin',
                        weights, '.05', '2026', probe_path], ROOT / f'{scene}_probe.log', scene)
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == old_sha
    probes.append(dict(scene=scene, **json.loads(probe_path.read_text())))
    output = ROOT / scene
    output.mkdir(exist_ok=False)
    run('train_worker_fixed', [source / 'training_manifest.json', source / 'training_input.bin',
                              output, '.05', '2026', '100', '.0001'], output / 'training.log', scene)
    records = [json.loads(x[3:]) for x in (output / 'training.log').read_text().splitlines() if x.startswith('PG ')]
    assert records[-1]['completed'] and records[-1]['frozen_exact']
    assert records[-1]['selection_gradient'] == 'implicit_expected_K_VJP'
    tr = [x for x in records if x['kind'] == 'training']
    hard = [x for x in records if x['kind'] == 'hard_summary']
    assert len(tr) == 100
    legacy_records = json.loads((source / 'legacy_models/PG_5_seed_2026/completed.json').read_text())['records']
    assert [x['frame'] for x in tr] == legacy_records[-1]['visited_Bank_frames'][:100]
    snapshot = lambda step: dict(torch.jit.load(str(output / f'step_{step}.pt'), map_location='cpu').named_buffers())
    initial, final = snapshot(0), snapshot(100)
    old = dict(torch.jit.load(str(source / 'legacy_models/PG_5_seed_2026/step_2000.pt'), map_location='cpu').named_buffers())
    assert torch.equal(initial['cached_uniform'], old['cached_uniform'])
    assert torch.all(initial['correction'] == 0)
    data = np.load(source / 'ranking_inputs.npz')
    base = np.log(data['raw_I'].astype(float) + 1e-12)
    raw_order = np.lexsort((data['row_ids'], -base))
    np.testing.assert_array_equal(raw_order, np.load(source / 'rankings.npz')['A_order'])
    assert all(np.isfinite([x['sampled_photo'], x['theta_grad_norm'], x['score_grad_sum']]).all() for x in tr)
    normalized_sum = max(abs(x['score_grad_sum']) / (x['score_grad_abs_sum'] + 1e-30) for x in tr)
    assert normalized_sum < 1e-12
    assert max(abs(x['correction_mean']) for x in tr) < 1e-12
    model = torch.nn.Sequential(torch.nn.Linear(3,32),torch.nn.SiLU(),torch.nn.Linear(32,32),
                                torch.nn.SiLU(),torch.nn.Linear(32,1,bias=False)).double()
    model.load_state_dict(dict(torch.jit.load(str(output / 'step_100_weights.pt'), map_location='cpu').named_parameters()))
    with torch.no_grad():
        r = model(final['normalized_xyz']).flatten()
        torch.testing.assert_close(r-r.mean(), final['correction'], atol=1e-9, rtol=1e-10)
    score = base + final['correction'].numpy()
    order = np.lexsort((data['row_ids'], -score))
    k = int(len(base)*.05 + .5)
    np.testing.assert_array_equal(np.sort(data['row_ids'][order[:k]]), hard[-1]['selected_source_rows'])
    result = dict(scene=scene, numerical_checks_passed=True, steps=100, lr=.0001,
                  initial_Bank_photo=hard[0]['photo'], final_Bank_photo=hard[-1]['photo'],
                  Bank_photo_improved=hard[-1]['photo']<hard[0]['photo'],
                  score_gradient_normalized_sum_max=normalized_sum,
                  correction_mean_abs_max=max(abs(x['correction_mean']) for x in tr),
                  correction_std_final=tr[-1]['correction_std'],
                  frozen_exact=True, cached_uniform_and_Bank_schedule_exact=True,
                  independent_weight_forward_and_hard_ranking_exact=True)
    results.append(result)
    print(json.dumps(result), flush=True)

summary = dict(all_correctness_checks_passed=True, regression=regression, native_probes=probes,
               Bank_sanity=results, formal_training_ready=all(x['Bank_photo_improved'] for x in results),
               new_training_updates=300, evaluation_renders=0, LR_tuning=False,
               fixed_source_SHA=hashlib.sha256((ROOT / 'source/train_worker.cpp').read_bytes()).hexdigest())
(ROOT / 'verification.json').write_text(json.dumps(summary, indent=2) + '\n')
