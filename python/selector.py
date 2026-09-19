"""Selector inference helpers shared by the LOD evaluation script."""

import math
import os

import numpy as np
import torch
from torch import nn
from plyfile import PlyData


class SelectorNetwork(nn.Module):
    """Python counterpart of improvements::selector::GumbelNetwork."""

    embedding_width = 32

    def __init__(self):
        super().__init__()
        self.pos_emd = self._make_embedding(3)
        self.rotation_emd = self._make_embedding(4)
        self.scale_emd = self._make_embedding(3)
        self.time_emd = self._make_embedding(1)
        self.soft_net = nn.Linear(self.embedding_width * 4, 2)

    @classmethod
    def _make_embedding(cls, input_width):
        return nn.Sequential(
            nn.Linear(input_width, cls.embedding_width),
            nn.ReLU(),
            nn.Linear(cls.embedding_width, cls.embedding_width),
            nn.ReLU(),
        )

    def forward(self, position, rotation, scale, target_ratio, tau=1.0):
        features = torch.cat(
            (
                self.pos_emd(position),
                self.rotation_emd(rotation),
                self.scale_emd(scale),
                self.time_emd(target_ratio),
            ),
            dim=1,
        )
        logits = self.soft_net(features)
        return torch.softmax(logits / tau, dim=1)[:, 1]


def load_selector(path, device):
    """Load a selector checkpoint produced by the C++ or Python path."""
    if not os.path.isfile(path):
        raise FileNotFoundError("selector checkpoint not found: {}".format(path))

    model = SelectorNetwork().to(device)
    try:
        payload = torch.jit.load(path, map_location=device).state_dict()
    except RuntimeError:
        payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise RuntimeError(
            "selector.pt does not contain a readable selector checkpoint."
        )

    state = {}
    for key, value in payload.items():
        key = key.replace("module.", "", 1) if key.startswith("module.") else key
        state[key] = value
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "selector checkpoint keys do not match the selector network: "
            "missing={}, unexpected={}".format(missing, unexpected)
        )
    model.eval()
    return model


def load_selector_metadata(path, device):
    """Load the sidecar metadata PLY written by GaussianModel."""
    if not os.path.isfile(path):
        raise FileNotFoundError("selector metadata not found: {}".format(path))
    vertex = PlyData.read(path)["vertex"]
    fields = vertex.data.dtype.names
    required = ("selector_birth_iter", "selector_seen_count")
    missing = [name for name in required if name not in fields]
    if missing:
        raise RuntimeError("selector metadata is missing fields: {}".format(missing))
    birth = torch.as_tensor(
        np.asarray(vertex["selector_birth_iter"], dtype=np.int32),
        device=device,
        dtype=torch.int32,
    )
    seen = torch.as_tensor(
        np.asarray(vertex["selector_seen_count"], dtype=np.int32),
        device=device,
        dtype=torch.int32,
    )
    return birth, seen


@torch.no_grad()
def build_selector_mask(
    model,
    gaussians,
    birth_iter,
    seen_count,
    current_iteration,
    ratio,
    min_age,
    min_seen,
    temperature=1.0,
    protection_enabled=True,
):
    """Return the selector mask, optionally applying maturity protection."""
    total = gaussians.get_xyz.shape[0]
    if protection_enabled and (
        birth_iter is None
        or seen_count is None
        or birth_iter.numel() != total
        or seen_count.numel() != total
    ):
        raise RuntimeError(
            "selector metadata is required and must match Gaussian count {}".format(
                total
            )
        )

    target_ratio = torch.full(
        (total, 1), float(ratio), dtype=torch.float32, device=gaussians.get_xyz.device
    )
    scores = model(
        gaussians.get_xyz,
        gaussians.get_rotation,
        gaussians.get_scaling,
        target_ratio,
        tau=temperature,
    )
    if protection_enabled:
        mature = (current_iteration - birth_iter >= min_age) & (
            seen_count >= min_seen
        )
        protected = ~mature
    else:
        mature = torch.ones(total, dtype=torch.bool, device=gaussians.get_xyz.device)
        protected = torch.zeros_like(mature)
    eligible_indices = torch.nonzero(mature, as_tuple=False).flatten()
    eligible_count = int(eligible_indices.numel())
    selected_count = int(math.floor(float(ratio) * eligible_count))

    mask = protected.clone()
    if selected_count > 0:
        eligible_scores = scores[eligible_indices]
        selected = eligible_indices[
            torch.topk(eligible_scores, selected_count).indices
        ]
        mask[selected] = True

    stats = {
        "total_gaussians": int(total),
        "protected_gaussians": int(protected.sum().item()),
        "mature_gaussians": eligible_count if protection_enabled else 0,
        "eligible_gaussians": eligible_count,
        "selected_mature_gaussians": selected_count,
        "selected_eligible_gaussians": selected_count,
        "active_gaussians": int(mask.sum().item()),
        "ratio": float(ratio),
        "protection_enabled": bool(protection_enabled),
        "current_iteration": int(current_iteration),
        "min_age": int(min_age),
        "min_seen": int(min_seen),
    }
    return mask.to(dtype=torch.float32), stats
