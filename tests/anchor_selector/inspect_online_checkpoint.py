"""复核真实在线快照的行关联、固定成熟度及 native/Python 四 ratio 一致性。"""
import json
from pathlib import Path
import sys
import torch

repo, checkpoint, probe_path, output = map(Path, sys.argv[1:5])
sys.path.insert(0, str(repo/"python"))
from selector import load_selector, load_selector_metadata, build_selector_mask
from gaussian_model import GaussianModel
from elastic_budget import DEFAULTS

model = load_selector(str(checkpoint/"selector.pt"), "cuda")
iteration = checkpoint.parent.name.split("_")[0]
ply = checkpoint/"point_cloud"/("iteration_"+iteration)
gaussians = GaussianModel(3); gaussians.load_ply(str(ply/"point_cloud.ply"))
metadata = load_selector_metadata(str(ply/"selector_metadata.ply"), "cuda")
probe = torch.jit.load(str(probe_path), map_location="cuda").state_dict()
assert (metadata["maturity"] == 1).all() and (metadata["drift_ema"] == 0).all()
assert all(torch.isfinite(p).all() for p in model.parameters())
ids = model.anchors.rows
assert ids.numel() == gaussians.get_xyz.shape[0]
occupancy = torch.bincount(ids, minlength=model.anchors.features.shape[0])
assert (occupancy > 0).all()
results = []
for i, ratio in enumerate((.01,.05,.1,.15)):
    with torch.no_grad():
        probability = model(gaussians.get_xyz, None, None, torch.tensor([[ratio]], device="cuda"))
        error = float((probability-probe["probability_"+str(i)]).abs().max())
        mask, stats = build_selector_mask(model, gaussians, metadata, ratio, DEFAULTS, protection_enabled=False)
        assert torch.equal(mask.bool(), probe["mask_"+str(i)])
        # Native LibTorch and Python PyTorch use different releases; allow FP32
        # rounding in probabilities, but keep the actual Bernoulli mask exact.
        assert error < 1e-6, error
        results.append(dict(ratio=ratio, actual_count=int(mask.sum()),
                            nominal_floor_k=int(ratio*ids.numel()), expected_count=stats["expected_active"],
                            max_probability_error=error, native_mask_exact=True))
summary = dict(passed=True, population=ids.numel(), anchors=occupancy.numel(), maturity_fixed_one=True,
               finite_parameters=True, feature_nonzero_cell_fraction=float((model.anchors.features.abs().sum(1)>0).float().mean()),
               occupancy_min=int(occupancy.min()), occupancy_median=float(occupancy.float().median()),
               occupancy_max=int(occupancy.max()), ratios=results)
output.write_text(json.dumps(summary, indent=2)); print(json.dumps(summary, indent=2))
