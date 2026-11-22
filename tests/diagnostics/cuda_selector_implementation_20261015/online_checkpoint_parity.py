"""在线完成后，逐一对同一个实际 checkpoint 做 Python inference/mask/RNG A/B。"""
import ast
import csv
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

dest = Path(__file__).resolve().parent
root = dest.parents[2]
output = root/'results/selector_online_ab_final_20261015'
while True:
    try:
        state = json.loads((output/'status.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        time.sleep(1)
        continue
    if state.get('error'):
        raise RuntimeError(state['error'])
    if state.get('complete'):
        break
    time.sleep(5)

import cv2
import numpy as np
from plyfile import PlyData
import torch

sys.path.insert(0, str(root/'python'))
from selector import load_selector, load_selector_metadata, build_selector_mask

# Reuse the actual config parser without importing run.py's global LPIPS models.
tree = ast.parse((root/'python/run.py').read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_selector_config')
namespace = dict(os=os, cv2=cv2)
exec(compile(ast.Module(body=[function], type_ignores=[]), 'run.py:load_selector_config', 'exec'), namespace)
read_config = namespace['load_selector_config']
torch.manual_seed(0)
failures = 0
fields = ['run','ratio','N','selected','max_abs','rms','p99','worst_global_gaussian_id','unequal_probabilities',
          'probability_within_tolerance','current_mask_exact','compact_ids_exact','seed0_bernoulli_exact',
          'seed0_next_draw_exact','seed0_state_exact','two_sided_near_tie_exact','global_rng_unchanged','passed']
with (dest/'online_checkpoint_parity.csv').open('w', newline='') as stream, torch.no_grad():
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for run in state['runs']:
        checkpoints = list((output/run['name']).glob('*_shutdown/ply'))
        assert len(checkpoints) == 1, checkpoints
        checkpoint = checkpoints[0]
        cloud = next((checkpoint/'point_cloud').glob('iteration_*/point_cloud.ply'))
        vertex = PlyData.read(cloud)['vertex']
        def tensor(names):
            return torch.from_numpy(np.stack([vertex[n] for n in names], axis=1).astype(np.float32)).cuda()
        xyz = tensor(['x','y','z'])
        rotation = torch.nn.functional.normalize(tensor([f'rot_{i}' for i in range(4)]), dim=1)
        scale = tensor([f'scale_{i}' for i in range(3)]).exp()
        gaussians = SimpleNamespace(get_xyz=xyz, get_rotation=rotation, get_scaling=scale)
        metadata = load_selector_metadata(cloud.parent/'selector_metadata.ply', 'cuda')
        config = read_config(str(checkpoint/'selector_config.yaml'))
        model = load_selector(str(checkpoint/'selector.pt'), 'cuda')
        n = len(vertex)
        for ratio in (.01,.05,.10,.15):
            target = torch.full((n,1),ratio,device='cuda')
            rng = torch.cuda.get_rng_state()
            model.set_backend('legacy')
            a = model(xyz,rotation,scale,target,tau=config['temperature'])
            ma,_ = build_selector_mask(model,gaussians,metadata,ratio,config,temperature=config['temperature'],
                                       protection_enabled=bool(config['protection_enabled']))
            model.set_backend('cuda')
            b = model(xyz,rotation,scale,target,tau=config['temperature'])
            mb,_ = build_selector_mask(model,gaussians,metadata,ratio,config,temperature=config['temperature'],
                                       protection_enabled=bool(config['protection_enabled']))
            # The frozen original evaluation rule draws N uniforms with seed0.
            # Test that independent sampler separately from the current caller.
            ga = torch.Generator(device='cuda').manual_seed(0)
            gb = torch.Generator(device='cuda').manual_seed(0)
            original_a = torch.rand(n,device='cuda',generator=ga) < a
            original_b = torch.rand(n,device='cuda',generator=gb) < b
            next_exact = torch.equal(torch.rand(32,device='cuda',generator=ga),torch.rand(32,device='cuda',generator=gb))
            below = torch.nextafter(a,torch.full_like(a,-float('inf')))
            near = torch.equal(a<a,a<b) and torch.equal(below<a,below<b)
            error = (a-b).abs()
            maximum = float(error.max())
            row = dict(run=run['name'],ratio=ratio,N=n,selected=int(ma.sum()),max_abs=maximum,
                rms=float(error.square().mean().sqrt()),p99=float(error.quantile(.99)),
                worst_global_gaussian_id=int(error.argmax()) if maximum else -1,
                unequal_probabilities=int((a!=b).sum()),probability_within_tolerance=torch.allclose(a,b,rtol=2e-4,atol=2e-6),
                current_mask_exact=torch.equal(ma,mb),compact_ids_exact=torch.equal(torch.nonzero(ma).flatten(),torch.nonzero(mb).flatten()),
                seed0_bernoulli_exact=torch.equal(original_a,original_b),seed0_next_draw_exact=next_exact,
                seed0_state_exact=torch.equal(ga.get_state(),gb.get_state()),
                two_sided_near_tie_exact=near,global_rng_unchanged=torch.equal(rng,torch.cuda.get_rng_state()))
            row['passed'] = all(row[k] for k in fields[9:-1])
            failures += not row['passed']
            writer.writerow(row)
            stream.flush()
        print(run['name']+' actual checkpoint inference parity complete',flush=True)
(dest/'online_checkpoint_parity.exit').write_text(str(int(bool(failures)))+'\n')
print(f'Actual online checkpoints: {len(state["runs"])}; failures={failures}')
raise SystemExit(bool(failures))
