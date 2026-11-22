"""对 native checkpoint 的四 ratio 概率、maturity 混合和 cached mask 做独立复核。"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from selector import load_selector, build_selector_mask, AnchorSelectorNetwork
from elastic_budget import DEFAULTS

device = sys.argv[2] if len(sys.argv) > 2 else "cuda"
directory = Path(sys.argv[1])
model = load_selector(str(directory / "selector_test.pt"), device)
assert isinstance(model, AnchorSelectorNetwork)
probe = torch.jit.load(str(directory / "selector_probe.pt"), map_location=device).state_dict()
metadata = dict(maturity=probe["maturity"], drift_ema=torch.zeros_like(probe["maturity"]),
                cached_uniform=probe["uniform"], mapping_iteration=1)
gaussians = SimpleNamespace(get_xyz=probe["xyz"])
errors = []
for i, ratio in enumerate((.01, .05, .1, .15)):
    with torch.no_grad():
        p = model(gaussians.get_xyz, None, None, torch.tensor([[ratio]], device=device))
        error = float((p-probe["probability_"+str(i)]).abs().max())
        assert error < 2e-7, error
        mask, stats = build_selector_mask(model, gaussians, metadata, ratio, DEFAULTS, temperature=1)
        assert torch.equal(mask.bool(), probe["mask_"+str(i)])
        errors.append(dict(ratio=ratio, max_probability_error=error,
                           actual_count=int(mask.sum()), expected_count=stats["expected_active"]))
print(json.dumps(dict(passed=True, device=device, ratios=errors)))
