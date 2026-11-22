"""Selector inference helpers shared by the LOD evaluation script."""

import os
import json

import numpy as np
import torch
from torch import nn
from plyfile import PlyData
from elastic_budget import classify_protection, protection_budget, protection_statistics


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
    """Load evidence, cached uniforms and the normal Mapping clock."""
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    fields = vertex.data.dtype.names
    comments = dict(line.split(" ", 1) for line in ply.comments if " " in line)
    if comments.get("protection_schema") != "evidence_aware":
        raise RuntimeError("checkpoint does not contain evidence-aware Protection metadata")
    width = int(comments["protection_unique_keyframe_ready"])
    def ids(prefix):
        low = np.asarray(vertex[prefix+"_low"], dtype=np.uint32).astype(np.uint64)
        high = np.asarray(vertex[prefix+"_high"], dtype=np.uint32).astype(np.uint64)
        return torch.as_tensor((low | (high << 32)).view(np.int64).copy(), device=device)
    metadata = dict(birth_iter=ids("birth_iter"),
        optimized_count=torch.as_tensor(np.asarray(vertex["optimized_count"], dtype=np.int32).copy(), device=device),
        unique_keyframe_count=torch.as_tensor(np.asarray(vertex["unique_keyframe_count"], dtype=np.int32).copy(), device=device),
        supporting_keyframe_ids=torch.stack([ids("supporting_keyframe_id_"+str(k)) for k in range(width)], 1),
        cached_uniform=torch.as_tensor(np.asarray(vertex["cached_uniform"], dtype=np.float32).copy(), device=device),
        mapping_iteration=int(comments["protection_mapping_iteration"]),
        unique_keyframe_ready=width)
    state_path = os.path.join(os.path.dirname(path), "protection_state.json")
    if os.path.isfile(state_path):
        with open(state_path, encoding="utf-8") as stream:
            state = json.load(stream)
        for key, value in state.items():
            if key.endswith("_to_mature") or key in ("warmup_to_under_observed", "under_observed_to_stale"):
                metadata[key] = value
    return metadata

@torch.no_grad()
def build_selector_mask(model, gaussians, metadata, ratio, config, temperature=1.0,
                        protection_enabled=True):
    """Condition the Mature Selector on global ratio; reserve actual Protection admissions."""
    total = gaussians.get_xyz.shape[0]
    if metadata is None or metadata["birth_iter"].numel() != total:
        raise RuntimeError("evidence-aware metadata must match Gaussian count")
    if metadata["unique_keyframe_ready"] != config["unique_keyframe_ready"]:
        raise RuntimeError("Protection KF threshold differs from checkpoint support width")
    c = classify_protection(metadata, config, protection_enabled)
    mature = c["mature"]
    eligible = torch.nonzero(mature, as_tuple=False).flatten()
    mask, probability, budget = protection_budget(
        mature, c["need"], metadata["cached_uniform"], ratio, config["max_budget_fraction"])
    if eligible.numel():
        target_ratio = torch.full((eligible.numel(), 1), float(ratio),
                                  dtype=torch.float32, device=gaussians.get_xyz.device)
        scores = model(gaussians.get_xyz[eligible], gaussians.get_rotation[eligible],
                       gaussians.get_scaling[eligible], target_ratio, tau=1.0)
        # Independent draw from the Selector's categorical keep distribution.
        # Protection draws use the saved uniforms and are unchanged across ratios.
        generator = torch.Generator(device=gaussians.get_xyz.device).manual_seed(0)
        draw = torch.rand(total, device=gaussians.get_xyz.device, generator=generator)
        selected = eligible[draw[eligible] < scores]
        mask[selected] = True
    else:
        selected = eligible
    active = int(mask.sum().item())
    stats = protection_statistics(metadata, c)
    stats.update(budget, total_gaussians=int(total), mature_actual=int(selected.numel()),
        active_gaussians=active, actual_vs_target=active/max(1.0, budget["target_total_gaussians"]),
        ratio=float(ratio), protection_enabled=bool(protection_enabled),
        current_mapping_iteration=metadata["mapping_iteration"])
    return mask.float(), stats
