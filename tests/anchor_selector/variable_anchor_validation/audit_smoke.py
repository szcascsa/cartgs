"""独立检查真实训练的成员数、交替事件、调度以及 native/Python 导出一致性。"""
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
import torch

root = Path('/home/prmi/zwj/variable-anchor-20261021')
repo = root/'source'
out = root/'validation'
run = root/'variable_seed2026'
cp = run/'10000_shutdown/ply'
cloud = cp/'point_cloud/iteration_10000'
sys.path.insert(0, str(repo/'python'))
from selector import load_selector, load_selector_metadata, build_selector_mask
from gaussian_model import GaussianModel

gates = json.loads((out/'correctness.json').read_text())
assert len(gates) == 11 and all(g['passed'] and g['code'] == 0 for g in gates.values())
assert json.loads((out/'smoke_result.json').read_text())['exit_code'] == 0
model = load_selector(str(cp/'selector.pt'), 'cuda')
counts = model.anchors.member_counts
owners = model.anchors.rows
assert ((counts >= 1)&(counts <= 10)).all()
assert torch.equal(torch.bincount(owners, minlength=len(counts)), counts)
assert len(owners) == int(counts.sum())
assert model.local_emd[0].in_features == 27
histogram = torch.bincount(counts, minlength=11).cpu().tolist()
levels = torch.bincount(model.anchors.levels, minlength=3).cpu().tolist()
text = (run/'training.log').read_text()
events = re.findall(r'Alternating Topology\] round=(\d+) phase=(Gaussian|Anchor) mapping=(\d+)', text)
assert events and [int(e[0]) for e in events] == list(range(len(events)))
assert all(phase == ('Gaussian' if int(i)%2 == 0 else 'Anchor') for i, phase, _ in events)
assert len(events) == int(model.anchors.topology_round)
state = json.loads((cp/'ultrafast_rotation_state.json').read_text())
assert state['serial_full_lod'] and not state['serial_lod_detach_master'] and state['post_topology_full_steps'] == 0
assert state['full_render_count'] == state['lod_render_count'] == state['ultrafast_total_normal_steps'] == 10000
stages = json.loads((cp/'mapping_stage_state.json').read_text())
assert sum(stages[p]['gaussian_updates'] for p in ['online', 'tail']) == 20000
assert sum(stages[p]['lod_nonfinite'] for p in ['online', 'tail']) == 0
trace = list(csv.DictReader((run/'serial_full_lod_trace.csv').open()))
assert len(trace) == 10000
for j, row in enumerate(trace):
    assert int(row['mapping']) == j+1 and int(row['ratio_index']) == j%4
    assert all(int(row[k]) == 1 for k in ['full', 'lod', 'full_gaussian_update', 'lod_gaussian_update', 'selector_update'])
config_text = (run/'input_selector.yaml').read_text()
assert 'GlobalGI.mode: "disabled"' in config_text
g = GaussianModel(3)
g.load_ply(str(cloud/'point_cloud.ply'))
metadata = load_selector_metadata(str(cloud/'selector_metadata.ply'), 'cuda')
assert (metadata['maturity'] == 1).all() and len(g.get_xyz) == len(owners)
with (out/'real_checkpoint_native.log').open('w') as log:
    subprocess.run([str(repo/'bin/anchor_checkpoint_probe'), str(cp/'selector.pt'), str(cloud/'point_cloud.ply'),
                    str(cloud/'selector_metadata.ply'), str(out/'real_checkpoint_native.pt'), 'explicit_blocks'],
                   stdout=log, stderr=subprocess.STDOUT, check=True)
fixture = torch.jit.load(str(out/'real_checkpoint_native.pt'), map_location='cuda').state_dict()
config = dict(drift_temperature=.1, beta_fast=.6, beta_slow=.15, compatibility_mature_threshold=.8, entry_maturity=0)
selection = []
for i, ratio in enumerate([.01, .05, .1, .15]):
    mask, _ = build_selector_mask(model, g, metadata, ratio, config, 1., False)
    assert torch.equal(mask.bool(), fixture['mask_'+str(i)])
    alpha = g.get_opacity.flatten()
    selection.append(dict(ratio=ratio, selected=int(mask.sum()), selected_below_cutoff=int(((alpha < 1/255)&mask.bool()).sum())))
history = torch.jit.load(str(cloud/'anchor_pruning_state.pt'), map_location='cpu').state_dict()
assert torch.equal(history['member_counts'], counts.cpu()) and torch.equal(history['gaussian_owner'], owners.cpu())
assert len(history['slot_current_opacity']) == len(owners)
created = sum(map(int, re.findall(r'Variable Anchor Insert\] anchors=(\d+)', text)))
clones = sum(map(int, re.findall(r'Variable Gaussian Densify\] clones=(\d+)', text)))
splits = sum(map(int, re.findall(r'Variable Gaussian Densify\].*?split_parents=(\d+)', text)))
source_manifest = json.loads((root/'source_manifest.json').read_text())
assert all(hashlib.sha256((repo/p).read_bytes()).hexdigest() == h for p, h in source_manifest.items())
result = dict(passed=True, correctness_gates=11, real_mapping_steps=len(trace), gaussian_updates=20000,
              anchors=len(counts), gaussians=len(owners), occupancy_histogram_0_to_10=histogram,
              anchors_by_level=levels, occupancy_mean=float(counts.float().mean()), created_anchors=created,
              removed_anchors=created-len(counts), clone_parents=clones, split_parents=splits,
              topology_events=len(events), gaussian_events=sum(p == 'Gaussian' for _, p, _ in events),
              anchor_events=sum(p == 'Anchor' for _, p, _ in events),
              native_python_cached_bernoulli_masks_exact=True, selection=selection,
              selector_forward_ms=fixture['native_forward_ms'].cpu().tolist(),
              quality_improvement_established=False, source_sha256_verified=len(source_manifest))
(out/'audit_summary.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
