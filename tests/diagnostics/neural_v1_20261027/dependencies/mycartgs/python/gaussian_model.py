#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import torch
from torch import nn
from plyfile import PlyData, PlyElement


class GaussianModel:
    def __init__(self, sh_degree: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.opacity_activation = torch.sigmoid
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.rotation_activation = torch.nn.functional.normalize

        # self.inverse_opacity_activation = inverse_sigmoid
        # self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    @torch.no_grad()
    def save_ply(self, path, selector_mask=None, elastic_attributes=None):
        """Write selected, compensated Gaussians in the standard binary 3DGS format."""
        xyz = self._xyz if elastic_attributes is None else elastic_attributes["xyz"]
        scaling = self._scaling if elastic_attributes is None else torch.log(elastic_attributes["scaling"])
        rotation = self._rotation if elastic_attributes is None else elastic_attributes["rotation"]
        appearance = elastic_attributes or {}
        dc = appearance.get("features_dc", self._features_dc)
        attributes = {
            "": xyz,
            "n": torch.zeros_like(xyz),
            "f_dc_": dc.transpose(1, 2).flatten(1),
            "f_rest_": self._features_rest.transpose(1, 2).flatten(1),
            "opacity": appearance.get("opacity_logits", self._opacity),
            "scale_": scaling,
            "rot_": rotation,
        }
        total = self._xyz.shape[0]
        if any(value.shape[0] != total for value in attributes.values()):
            raise ValueError("export attributes must match Gaussian count")
        active_mask = None
        if selector_mask is not None:
            if selector_mask.ndim != 1 or selector_mask.shape[0] != total:
                raise ValueError("selector mask must have one value per Gaussian")
            active_mask = selector_mask.to(device=xyz.device, dtype=torch.bool)
        columns = {}
        for prefix, value in attributes.items():
            if active_mask is not None:
                value = value[active_mask]
            array = value.detach().cpu().numpy()
            for index in range(array.shape[1]):
                name = prefix + "xyz"[index] if prefix in ("", "n") else prefix + str(index)
                if prefix == "opacity":
                    name = "opacity"
                columns[name] = array[:, index]
        count = len(columns["x"])
        vertices = np.empty(count, dtype=[(name, "<f4") for name in columns])
        for name, values in columns.items():
            vertices[name] = values
        PlyData([PlyElement.describe(vertices, "vertex")], text=False, byte_order="<").write(str(path))
        return count

    def load_ply(self, path, device="cuda"):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(
            extra_f_names, key=lambda x: int(x.split("_")[-1])
        )
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device=device).requires_grad_(
                True
            )
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device=device)
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device=device)
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(
                opacities, dtype=torch.float, device=device
            ).requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            torch.tensor(
                scales, dtype=torch.float, device=device
            ).requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(
                rots, dtype=torch.float, device=device
            ).requires_grad_(True)
        )

        self.active_sh_degree = self.max_sh_degree
