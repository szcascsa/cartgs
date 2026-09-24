"""Inference counterpart of the C++ canonical Transform Field."""

import json
import os

import numpy as np
import torch
import torch.nn.functional as functional
from plyfile import PlyData
from torch import nn


def _normalize_quaternion(quaternion):
    return quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _multiply_quaternion(lhs, rhs):
    scalar = lhs[:, :1] * rhs[:, :1] - (lhs[:, 1:] * rhs[:, 1:]).sum(
        dim=-1, keepdim=True
    )
    vector = (
        lhs[:, :1] * rhs[:, 1:]
        + rhs[:, :1] * lhs[:, 1:]
        + torch.cross(lhs[:, 1:], rhs[:, 1:], dim=-1)
    )
    return torch.cat((scalar, vector), dim=-1)


def _rotate(quaternion, vector):
    quaternion = _normalize_quaternion(quaternion)
    twice_cross = 2.0 * torch.cross(quaternion[:, 1:], vector, dim=-1)
    return (
        vector
        + quaternion[:, :1] * twice_cross
        + torch.cross(quaternion[:, 1:], twice_cross, dim=-1)
    )


class HexPlaneField(nn.Module):
    plane_names = ("xy", "xz", "xe", "yz", "ye", "ze")
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))

    def __init__(self, state, device):
        super().__init__()
        self.multires = tuple(state["multires"])
        self.feature_dim = int(state["feature_dim"])
        self.register_buffer(
            "aabb_min", torch.tensor(state["aabb_min"], dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "aabb_max", torch.tensor(state["aabb_max"], dtype=torch.float32, device=device)
        )
        spatial_resolution = int(state["spatial_resolution"])
        ratio_resolution = int(state["ratio_resolution"])
        for level, multiplier in enumerate(self.multires):
            resolution = (
                spatial_resolution * multiplier,
                spatial_resolution * multiplier,
                spatial_resolution * multiplier,
                ratio_resolution,
            )
            for name, (first, second) in zip(self.plane_names, self.pairs):
                self.register_parameter(
                    "L{}_{}".format(level, name),
                    nn.Parameter(
                        torch.empty(
                            1,
                            self.feature_dim,
                            resolution[second],
                            resolution[first],
                            device=device,
                        )
                    ),
                )

    def inside_mask(self, canonical_xyz):
        return ((canonical_xyz >= self.aabb_min) & (canonical_xyz <= self.aabb_max)).all(
            dim=1
        )

    def forward(self, canonical_xyz, ratio):
        if canonical_xyz.shape[0] == 0:
            return canonical_xyz.new_empty((0, self.feature_dim * len(self.multires)))
        normalized = 2.0 * (canonical_xyz - self.aabb_min) / (
            self.aabb_max - self.aabb_min
        ) - 1.0
        ratio_column = normalized.new_full((canonical_xyz.shape[0], 1), ratio)
        coordinates = torch.cat((normalized, ratio_column), dim=1)
        features = []
        for level in range(len(self.multires)):
            product = None
            for name, (first, second) in zip(self.plane_names, self.pairs):
                plane = getattr(self, "L{}_{}".format(level, name))
                grid = coordinates[:, (first, second)].reshape(
                    1, 1, canonical_xyz.shape[0], 2
                )
                sampled = functional.grid_sample(
                    plane, grid, mode="bilinear", padding_mode="border", align_corners=True
                ).reshape(self.feature_dim, -1).transpose(0, 1)
                product = sampled if product is None else product * sampled
            features.append(product)
        return torch.cat(features, dim=1)


class TransformField(nn.Module):
    def __init__(self, state, device):
        super().__init__()
        self.outside_identity = bool(state.get("outside_identity", True))
        self.mature_only = bool(state.get("mature_only", True))
        self.grid = HexPlaneField(state, device)
        hidden_dim = int(state["hidden_dim"])
        self.feature_out = nn.Linear(self.grid.feature_dim * len(self.grid.multires), hidden_dim)
        self.position_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.position_out = nn.Linear(hidden_dim, 3)
        self.scale_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.scale_out = nn.Linear(hidden_dim, 3)
        self.rotation_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.rotation_out = nn.Linear(hidden_dim, 4)
        self.to(device)

    def forward(self, canonical_xyz, ratio):
        feature = self.feature_out(self.grid(canonical_xyz, ratio))
        delta_xyz = self.position_out(
            functional.relu(self.position_hidden(functional.relu(feature)))
        )
        delta_scale = self.scale_out(
            functional.relu(self.scale_hidden(functional.relu(feature)))
        )
        delta_rotation = self.rotation_out(
            functional.relu(self.rotation_hidden(functional.relu(feature)))
        )
        return delta_xyz, delta_scale, delta_rotation


def load_canonical_frames(path, count, device):
    if not os.path.isfile(path):
        raise FileNotFoundError("canonical frames not found: {}".format(path))
    vertex = PlyData.read(path)["vertex"]
    columns = ("canonical_scale",) + tuple(
        "canonical_rot_{}".format(index) for index in range(4)
    ) + tuple("canonical_trans_{}".format(index) for index in range(3))
    if len(vertex) != count or not set(columns).issubset(vertex.data.dtype.names):
        raise RuntimeError("canonical frames do not align with the Gaussian PLY")
    values = torch.as_tensor(
        np.stack([vertex[column] for column in columns], axis=1).astype(np.float32),
        device=device,
    )
    if not torch.isfinite(values).all() or not (values[:, 0] > 0).all():
        raise RuntimeError("canonical frames contain invalid Sim(3) parameters")
    return values[:, :1], _normalize_quaternion(values[:, 1:5]), values[:, 5:]


def load_transform_field(ply_directory, device):
    state_path = os.path.join(ply_directory, "transform_state.json")
    if not os.path.isfile(state_path):
        return None
    with open(state_path, "r", encoding="utf-8") as stream:
        state = json.load(stream)
    if not state["initialized"]:
        return None
    path = os.path.join(ply_directory, "transform_field.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError("Transform Field checkpoint not found: {}".format(path))
    model = TransformField(state, device)
    try:
        payload = torch.jit.load(path, map_location=device).state_dict()
    except RuntimeError:
        payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise RuntimeError("Transform Field checkpoint is not a state dictionary")
    payload = {
        key.replace("module.", "", 1) if key.startswith("module.") else key: value
        for key, value in payload.items()
    }
    model.load_state_dict(payload, strict=True)
    model.eval()
    return model


@torch.no_grad()
def transformed_attributes(model, gaussians, frames, selected_mask, mature_mask, ratio):
    """Apply the field using the checkpoint's maturity/AABB policy."""
    world_xyz = gaussians.get_xyz.detach()
    if selected_mask.numel() != world_xyz.shape[0] or mature_mask.numel() != world_xyz.shape[0]:
        raise ValueError("selection and maturity masks must match Gaussian count")
    frame_scale, frame_rotation, frame_translation = frames
    if any(value.shape[0] != world_xyz.shape[0] for value in frames):
        raise ValueError("canonical frames must match Gaussian count")
    inverse_rotation = frame_rotation * frame_rotation.new_tensor((1, -1, -1, -1))
    canonical_xyz = _rotate(inverse_rotation, world_xyz - frame_translation) / frame_scale
    transform_mask = selected_mask.bool()
    if model.mature_only:
        transform_mask = transform_mask & mature_mask.bool()
    if model.outside_identity:
        transform_mask = transform_mask & model.grid.inside_mask(canonical_xyz)
    indices = torch.nonzero(transform_mask, as_tuple=False).flatten()
    if indices.numel() == 0:
        return None
    canonical = canonical_xyz[indices]
    delta_xyz, delta_log_scale, delta_rotation = model(canonical, float(ratio))
    local_scale = frame_scale[indices]
    local_rotation = frame_rotation[indices]
    local_translation = frame_translation[indices]
    xyz = world_xyz.clone()
    xyz[indices] = local_scale * _rotate(local_rotation, canonical + delta_xyz) + local_translation
    scaling = gaussians.get_scaling.detach().clone()
    scaling[indices] = scaling[indices] * torch.exp(delta_log_scale)
    rotation = gaussians.get_rotation.detach().clone()
    canonical_rotation = _multiply_quaternion(
        inverse_rotation[indices], rotation[indices]
    )
    identity = torch.zeros_like(delta_rotation)
    identity[:, 0] = 1.0
    rotated = _multiply_quaternion(
        _normalize_quaternion(identity + delta_rotation),
        _normalize_quaternion(canonical_rotation),
    )
    rotation[indices] = _normalize_quaternion(
        _multiply_quaternion(local_rotation, _normalize_quaternion(rotated))
    )
    return {"xyz": xyz, "scaling": scaling, "rotation": rotation}
