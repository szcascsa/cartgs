"""调用生产 evaluator；额外保存 mask，相同 K 仅作独立诊断。"""
import json
import os
from pathlib import Path
import runpy
import sys
import numpy as np
import torch

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo/"python"))
from evo.tools import settings
settings.SETTINGS.plot_backend = "Agg"
import selector

policy = os.environ["ANCHOR_SI_EVAL_POLICY"]
output = Path(sys.argv[sys.argv.index("--output_path")+1])
original = selector.build_selector_mask

@torch.no_grad()
def record(model, gaussians, metadata, ratio, config, temperature=1., protection_enabled=True):
    mask, stats = original(model, gaussians, metadata, ratio, config, temperature, protection_enabled)
    if policy == "same_k":
        n = gaussians.get_xyz.shape[0]
        k = json.loads(Path(os.environ["ANCHOR_SI_EVAL_PLAN"]).read_text())["common_k"][str(ratio)]
        assert (metadata["maturity"] == 1).all() and not protection_enabled
        extra = dict(sh_dc=gaussians.get_features[:, :1], opacity=gaussians.get_opacity) if getattr(model, "gaussian_attributes", False) else {}
        probability = model(gaussians.get_xyz, gaussians.get_rotation, gaussians.get_scaling,
                            torch.full((n, 1), ratio, device="cuda"), tau=temperature, **extra).cpu().numpy()
        order = np.lexsort((np.arange(n), -probability))
        stats["cached_bernoulli_count"] = stats["active_gaussians"]
        mask = torch.zeros(n, device="cuda")
        mask[torch.as_tensor(order[:k].copy(), device="cuda")] = 1
        stats.update(active_gaussians=k, actual_active=k, expected_active=float(k),
                     common_absolute_k=k, actual_vs_target=k/max(1., stats["target_active"]))
        assert int(mask.sum()) == k
    assert policy in ("cached_bernoulli", "same_k")
    stats["selection_policy"] = policy
    np.savez_compressed(output/"selected_rows.npz", rows=np.flatnonzero(mask.cpu().numpy()))
    return mask, stats

selector.build_selector_mask = record
runpy.run_path(str(repo/"python/run.py"), run_name="__main__")
