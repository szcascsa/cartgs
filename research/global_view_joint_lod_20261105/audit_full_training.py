"""核验实际训练预算、初始状态、相机序列及输入完整性，不启动GPU工作。"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
groups = ['baseline', 'joint', 'global_control']
reference = None; first_keys = None; master_hash = None; result = []


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))


for name in groups:
    folder = ROOT/'runs'/('full_fine_'+name)
    init = torch.jit.load(str(folder/'step0_selector.pt'), map_location='cpu').state_dict()
    if reference is None:
        reference = init
    assert set(reference) == set(init) and all(same(value, reference[key]) for key, value in init.items())
    digest = hashlib.sha256((folder/'step0_master.ply').read_bytes()).hexdigest()
    if master_hash is None:
        master_hash = digest
    assert master_hash == digest
    keys = (folder/'training_keyframes.csv').read_text()
    if first_keys is None:
        first_keys = keys
    assert keys == first_keys
    trajectory = folder/'CameraTrajectory_TUM.txt'
    metadata = folder/'23560_shutdown/ply/real_lod_state.json'
    state = json.loads(metadata.read_text())
    anchor = json.loads((metadata.parent/'point_cloud/iteration_23560/anchor_selector_state.json').read_text())
    rows = list(csv.DictReader((folder/'real_lod_steps.csv').open()))
    assert len(rows) == state['optimizer_commits'] == 11580
    assert state['ratio_steps'] == [1930]*6
    feature_clocks = [value['adam_step'] for value in anchor['compensation_feature_state']]
    assert feature_clocks == [11580, 9650, 7720, 5790, 3860, 1930]
    assert state['importance_full_renders'] == 864
    summary = {'group': name, 'steps': len(rows), 'master_updates': state['master_updates'],
               'selector_updates': state['selector_updates'], 'full_renders': state['full_training_renders'],
               'global_renders': state['lod_training_renders'], 'extra_renders': state['view_budget_training_renders'],
               'feature_adam_clocks': feature_clocks, 'initial_master_sha256': digest,
               'initial_selector_tensors': len(init), 'trajectory_sha256': hashlib.sha256(trajectory.read_bytes()).hexdigest(),
               'ratio_exposure': state['ratio_steps'], 'global_selected_total': sum(int(r['K']) for r in rows),
               'mean_mapping_ms': float(np.mean([float(r['ms']) for r in rows])),
               'median_mapping_ms': float(np.median([float(r['ms']) for r in rows])),
               'late_ratio': []}
    extras = [json.loads(line) for line in (folder/'view_budget_steps.jsonl').read_text().splitlines()] if name != 'baseline' else []
    if extras:
        assert len(extras) == 11580 and all(r['adam_commits_this_step'] == 1 and not r['strict_topk'] for r in extras)
        summary['extra_selected_total'] = sum(r['K'] for r in extras)
        assert all(abs(r['budget_loss']-abs(r['K']-r['target_K'])/max(1, r['target_K'])) < 1e-5 for r in extras)
    for ratio in [.01, .05, .1, .3, .5, .7]:
        recent = [r for r in rows if int(r['step']) > 9580 and abs(float(r['ratio'])-ratio) < 1e-6]
        value = {'ratio': ratio, 'global_actual_K_mean': float(np.mean([int(r['K']) for r in recent])),
                 'global_budget_relative_error_mean': float(np.mean([
                     abs(int(r['K'])-float(r['ratio'])*int(r['N']))/max(1, float(r['ratio'])*int(r['N'])) for r in recent]))}
        extra = [r for r in extras if r['step'] > 9580 and abs(r['ratio']-ratio) < 1e-6]
        if extra:
            value.update(extra_candidate_N_mean=float(np.mean([r['candidate_N'] for r in extra])),
                         extra_target_K_mean=float(np.mean([r['target_K'] for r in extra])),
                         extra_actual_K_mean=float(np.mean([r['K'] for r in extra])),
                         extra_budget_relative_error_mean=float(np.mean([r['budget_loss'] for r in extra])),
                         infeasible_budget_steps=sum(r['infeasible_budget_excess'] > 0 for r in extra),
                         zero_selection_steps=sum(r['K'] == 0 and r['candidate_N'] > 0 for r in extra))
        summary['late_ratio'].append(value)
    result.append(summary)
assert len({r['trajectory_sha256'] for r in result}) == 1
report = {'groups': result, 'same_initial_master': True, 'same_initial_selector_bitwise': True,
          'same_training_keyframe_sequence': True, 'same_final_trajectory': True, 'passed': True}
(ROOT/'reports/FULL_TRAINING_AUDIT.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
