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

import torch
import math
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from gaussian_model import GaussianModel


def render(
    viewpoint_camera,
    pc: GaussianModel,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    selector_mask=None,
    elastic_attributes=None,
):
    """Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor.
    # We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz,
            dtype=pc.get_xyz.dtype,
            requires_grad=True,
            device="cuda",
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz if elastic_attributes is None else elastic_attributes["xyz"]
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling if elastic_attributes is None else elastic_attributes["scaling"]
    rotations = pc.get_rotation if elastic_attributes is None else elastic_attributes["rotation"]
    if any(attribute.shape[0] != pc.get_xyz.shape[0] for attribute in (means3D, scales, rotations)):
        raise ValueError("elastic attributes must match Gaussian count")
    active_mask = None
    if selector_mask is not None:
        if selector_mask.ndim != 1 or selector_mask.shape[0] != opacity.shape[0]:
            raise ValueError("selector mask must have one value per Gaussian")
        # Zero opacity still sends every Gaussian through preprocessing.  The
        # rasterizer can then produce an all-invalid tile list when none of the
        # selected Gaussians contributes to a view.  Compact the inputs so
        # dropped Gaussians never enter the CUDA rasterizer.
        active_mask = selector_mask.to(device=means3D.device, dtype=torch.bool)
        means3D = means3D[active_mask].contiguous()
        means2D = means2D[active_mask].contiguous()
        opacity = opacity[active_mask].contiguous()

    # If precomputed 3d covariance is provided, use it.
    # If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None

    if active_mask is not None:
        scales = scales[active_mask].contiguous()
        rotations = rotations[active_mask].contiguous()

    # If precomputed colors are provided, use them.
    # Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it.
    # If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        dc, shs = pc.get_features_dc, pc.get_features_rest
        if active_mask is not None:
            dc = dc[active_mask].contiguous()
            shs = shs[active_mask].contiguous()
    else:
        colors_precomp = override_color
        if active_mask is not None:
            colors_precomp = colors_precomp[active_mask].contiguous()

    if active_mask is not None and not torch.any(active_mask):
        rendered_image = bg_color[:, None, None].expand(
            -1, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)
        ).clone()
        radii = torch.zeros(
            selector_mask.shape[0], dtype=torch.int32, device=means3D.device
        )
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        dc=dc,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    if active_mask is not None:
        compact_radii = radii
        radii = torch.zeros(
            selector_mask.shape[0],
            dtype=compact_radii.dtype,
            device=compact_radii.device,
        )
        radii[active_mask] = compact_radii

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
