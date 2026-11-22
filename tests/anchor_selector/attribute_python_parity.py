"""核对27维新布局与原20维布局，训练前后均可使用。"""
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"python"))
from selector import load_selector, build_selector_mask
from elastic_budget import DEFAULTS

folder, device = Path(sys.argv[1]), sys.argv[2]
model = load_selector(str(folder/"attribute_selector_test.pt"), device)
probe = torch.jit.load(str(folder/"attribute_selector_probe.pt"), map_location=device).state_dict()
gaussians = SimpleNamespace(get_xyz=probe["xyz"], get_scaling=probe["scale"],
                           get_features=probe["sh_dc"], get_opacity=probe["opacity"])
metadata = dict(maturity=torch.ones(len(probe["xyz"]),device=device),
                drift_ema=torch.zeros(len(probe["xyz"]),device=device),
                cached_uniform=probe["uniform"], mapping_iteration=1)
assert model.gaussian_attributes and int(model.anchor_layout_version) == 3
for i, ratio in enumerate((.01,.05,.1,.15)):
    with torch.no_grad():
        p = model(probe["xyz"], None, probe["scale"], torch.tensor([[ratio]],device=device),
                  sh_dc=probe["sh_dc"],opacity=probe["opacity"])
        error = float((p-probe["probability_"+str(i)]).abs().max())
        assert error < 5e-7, error
        mask, _ = build_selector_mask(model, gaussians, metadata, ratio, DEFAULTS, protection_enabled=False)
        assert torch.equal(mask.bool(), probe["uniform"] < probe["probability_"+str(i)])
print("PASS: native/Python 27-input probabilities, exact cached masks, scale/DC/opacity lookup")
