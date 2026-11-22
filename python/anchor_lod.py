"""可选的Anchor距离分层推理；名义格子裁剪，各档合并后统一光栅化。"""
import math
import time
import torch
from anchor_frustum import anchor_frustum_mask


def _thresholds(values, count, descending, name):
    values = tuple(float(x) for x in values)
    if (len(values) != count or any(not math.isfinite(x) or x <= 0 for x in values) or
            any((a <= b if descending else a >= b) for a, b in zip(values, values[1:]))):
        raise ValueError(f'{name} requires {count} positive finite, strictly '
                         f'{"decreasing" if descending else "increasing"} thresholds')
    return values


def schedule_levels(distance, screen_height, previous, distance_limits, screen_limits, hysteresis):
    """层级0为Full；只接收本帧非空且名义格子可见的Anchor。"""
    boundary = torch.arange(len(distance_limits), device=distance.device)
    known = previous[:, None] >= 0
    coarse = previous[:, None] > boundary
    distance_cut = distance_limits * torch.where(known, torch.where(coarse, 1-hysteresis, 1+hysteresis), 1.)
    distance_level = (distance[:, None] >= distance_cut).sum(1)
    if screen_limits is None:
        return distance_level
    screen_cut = screen_limits * torch.where(known, torch.where(coarse, 1+hysteresis, 1-hysteresis), 1.)
    screen_level = (screen_height[:, None] < screen_cut).sum(1)
    return torch.minimum(distance_level, screen_level)


@torch.no_grad()
def _master_bounds(gaussians, owners, anchor_count):
    """一次性构造3σ成员AABB，仅用于分级，不改变名义格子的可见性。"""
    xyz, scale, q = gaussians.get_xyz, gaussians.get_scaling, gaussians.get_rotation
    w, x, y, z = q.unbind(1)
    rotation = torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                            2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                            2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), 1).reshape(-1, 3, 3)
    extent = 3 * ((rotation * scale[:, None, :]).square().sum(2)).sqrt()
    low = xyz.new_full((anchor_count, 3), float('inf'))
    high = xyz.new_full((anchor_count, 3), -float('inf'))
    indices = owners[:, None].expand(-1, 3)
    low.scatter_reduce_(0, indices, xyz-extent, reduce='amin', include_self=True)
    high.scatter_reduce_(0, indices, xyz+extent, reduce='amax', include_self=True)
    empty = torch.bincount(owners, minlength=anchor_count) == 0
    low[empty], high[empty] = 0, 0
    return (low+high)*.5, (high-low)*.5


class MixedAnchorLodRenderer:
    """冻结checkpoint的有限档位缓存；prepare只分级/Gather，不更新模型或执行网络。"""
    def __init__(self, model, gaussians, config, distances, screen_thresholds=None,
                 hysteresis=.1, screen_guard=True, seed=0, temperature=1., protection_enabled=False):
        from selector import AnchorSelectorNetwork
        spatial = getattr(model, 'spatial_compensation', None)
        if (not isinstance(model, AnchorSelectorNetwork) or not model.explicit_blocks or
                config.get('selection_mode') != 'flexgs' or config.get('entry_maturity', 0.) != 0 or
                protection_enabled or getattr(model, 'reliability_gate_config', {}).get('enabled', False) or
                hasattr(model, 'gaussian_compensation') or spatial is None or
                spatial.ratio_feature_width != 48 or not spatial.offset_only or
                getattr(model.anchors, 'compensation_residual_mode', 0) != 1):
            raise ValueError('Mixed Anchor LoD requires Residual48 FlexGS, entry=0 and disabled maturity gates')
        if temperature != 1. or not math.isfinite(hysteresis) or not 0 <= hysteresis < .5:
            raise ValueError('Mixed LoD requires temperature=1 and hysteresis in [0,.5)')
        if not bool(spatial.normalization_ready):
            raise ValueError('Compensation normalization must be initialized in the checkpoint')
        self.model, self.gaussians = model, gaussians
        self.ratios = tuple(reversed(model.anchors.compensation_values))
        if (not 2 <= len(self.ratios) <= 6 or any(not math.isfinite(r) or not 0 < r <= 1 for r in self.ratios) or
                any(a <= b for a, b in zip(self.ratios, self.ratios[1:]))):
            raise ValueError('Expected two to six increasing trained ratio knots in the checkpoint')
        self.distances = _thresholds(distances, len(self.ratios), False, 'LoD distances')
        self.screen_guard, self.hysteresis, self.seed = bool(screen_guard), float(hysteresis), int(seed)
        self.screen_thresholds = _thresholds(
            screen_thresholds if screen_thresholds is not None else [.1 / 2**j for j in range(len(self.ratios))],
            len(self.ratios), True, 'LoD screen heights')
        self.labels = ('full',) + tuple('r' + format(r, '.6g').replace('.', 'p') for r in self.ratios)
        self.rebuild_cache()

    def _signature(self):
        tensors = [(n, getattr(self.gaussians, n)) for n in
                   ('_xyz', '_scaling', '_rotation', '_opacity', '_features_dc', '_features_rest')]
        tensors += list(self.model.named_parameters()) + list(self.model.named_buffers())
        return (self.gaussians.active_sh_degree, tuple(self.model.anchors.compensation_values), tuple(
            (n, id(t), t._version, t.data_ptr(), tuple(t.shape)) for n, t in tensors))

    def noise(self, level):
        generator = torch.Generator(device=self.gaussians.get_xyz.device).manual_seed(self.seed + level)
        return -torch.empty((self.total, 2), device=self.gaussians.get_xyz.device,
                            dtype=self.gaussians.get_xyz.dtype).exponential_(generator=generator).log()

    @torch.no_grad()
    def rebuild_cache(self):
        from selector import compensated_attributes, slot_eligibility
        from selector_sampling import flexgs_inference_mask
        from renderer import prepare_render_data
        started = time.perf_counter()
        pc, model = self.gaussians, self.model
        anchors = model.anchors
        if tuple(reversed(anchors.compensation_values)) != self.ratios:
            raise RuntimeError('Ratio knots changed; construct a new MixedAnchorLodRenderer with matching thresholds')
        self.total, self.anchor_count = len(pc.get_xyz), len(anchors.centers)
        self.owners = anchors.rows.detach().clone()
        if (self.owners.shape != (self.total,) or self.owners.dtype != torch.long or
                (self.total and (int(self.owners.min()) < 0 or int(self.owners.max()) >= self.anchor_count))):
            raise ValueError('Master Gaussian rows must align with Anchor ownership')
        if (anchors.offset_scales.shape != (self.anchor_count, 1) or
                not torch.isfinite(anchors.centers).all() or not torch.isfinite(anchors.offset_scales).all() or
                not (anchors.offset_scales > 0).all()):
            raise ValueError('Invalid nominal Anchor cells')
        full = prepare_render_data(pc)['inputs']
        if any(not torch.isfinite(full[n]).all() for n in ('means3D', 'scales', 'rotations', 'opacities', 'dc', 'shs')):
            raise ValueError('Nonfinite Master attributes cannot build a mixed LoD cache')
        self.occupied = torch.bincount(self.owners, minlength=self.anchor_count) > 0
        self.eligible = slot_eligibility(model, pc.get_opacity).clone()
        self.bound_centers, self.bound_half = _master_bounds(pc, self.owners, self.anchor_count)
        self.bound_radius = self.bound_half.norm(dim=1)
        self.distance_limits = pc.get_xyz.new_tensor(self.distances)
        self.screen_limits = pc.get_xyz.new_tensor(self.screen_thresholds) if self.screen_guard else None
        self.previous_levels = torch.full((self.anchor_count,), -1, device=pc.get_xyz.device, dtype=torch.long)
        self.masks = [torch.ones(self.total, device=pc.get_xyz.device, dtype=torch.bool)]
        pack = lambda a: torch.cat((a['means3D'], a['opacities'], a['scales'], a['rotations'], a['dc'].flatten(1)), 1)
        packed = [pack(full)]
        for level, ratio in enumerate(self.ratios, 1):
            if self.total:
                _, shared, logits = model.forward_selection_features(pc.get_xyz, pc.get_scaling,
                    pc.get_features_dc, pc.get_opacity, pc.get_xyz.new_tensor([[ratio]]), return_logits=True)
                mask, _, _ = flexgs_inference_mask(logits, ratio, eligible=self.eligible, noise=self.noise(level))
                attributes = compensated_attributes(model, pc, mask, ratio, shared)
                if attributes is not None:
                    # Match the existing evaluator's PLY scale/rotation round trip.
                    attributes['scaling'] = attributes['scaling'].log().exp()
                    attributes['rotation'] = pc.rotation_activation(attributes['rotation'])
                data = prepare_render_data(pc, elastic_attributes=attributes)['inputs']
            else:
                mask, data = self.masks[0], full
            self.masks.append(mask)
            packed.append(pack(data))
        self.masks = torch.stack(self.masks)
        self.attributes = torch.stack(packed)
        if not torch.isfinite(self.attributes).all():
            raise ValueError('Nonfinite Compensation attributes cannot build a mixed LoD cache')
        self.sh_rest = full['shs'].clone()
        self.level_ratios = pc.get_xyz.new_tensor((1.,) + self.ratios)
        if pc.get_xyz.is_cuda:
            torch.cuda.synchronize(pc.get_xyz.device)
        cache_tensors = (self.attributes, self.masks, self.sh_rest, self.owners, self.eligible, self.occupied,
            self.bound_centers, self.bound_half, self.bound_radius, self.previous_levels, self.level_ratios,
            self.distance_limits) + ((self.screen_limits,) if self.screen_limits is not None else ())
        self.cache_info = dict(levels=list(self.labels), ratios=[None] + list(self.ratios),
            distances=list(self.distances), screen_thresholds=list(self.screen_thresholds),
            screen_guard=self.screen_guard, hysteresis=self.hysteresis, sampling_seed=self.seed,
            full_gaussians=self.total, anchors=self.anchor_count,
            cache_bytes=sum(t.numel()*t.element_size() for t in cache_tensors),
            cache_build_ms=(time.perf_counter()-started)*1000,
            nominal_bounds='center +/- offsetScale/2; no padding',
            selection_policy='cached_full_map_flexgs_percentile_then_anchor_level',
            budget_policy='sum_actual_selected_members; no per-anchor or per-frame strict K')
        self._cached_signature = self._signature()

    @torch.no_grad()
    def _assemble(self, visible, levels):
        rows = torch.nonzero(visible[self.owners]).flatten()
        level = levels[self.owners[rows]]
        selected = rows[self.masks[level, rows]]
        chosen = levels[self.owners[selected]]
        value = self.attributes[chosen, selected]
        return {'sh_degree': self.gaussians.active_sh_degree, 'inputs': dict(
            means3D=value[:, :3].contiguous(), opacities=value[:, 3:4].contiguous(),
            scales=value[:, 4:7].contiguous(), rotations=value[:, 7:11].contiguous(),
            dc=value[:, 11:14].unsqueeze(1).contiguous(), shs=self.sh_rest[selected].contiguous(),
            means2D=torch.zeros_like(value[:, :3]), colors_precomp=None, cov3D_precomp=None)}, selected, rows

    @torch.no_grad()
    def prepare(self, camera):
        if self._signature() != self._cached_signature:
            raise RuntimeError('Mixed LoD cache invalidated by model/Master/topology change; call rebuild_cache()')
        visible = anchor_frustum_mask(self.model.anchors.centers, self.model.anchors.offset_scales,
                                      camera.full_proj_transform, self.occupied)
        ids = torch.nonzero(visible).flatten()
        center, half = self.bound_centers[ids], self.bound_half[ids]
        distance = ((center-camera.camera_center).abs()-half).clamp_min(0).norm(dim=1)
        camera_space = torch.cat((center, torch.ones_like(center[:, :1])), 1) @ camera.world_view_transform
        radius, depth = self.bound_radius[ids], camera_space[:, 2]
        screen = radius / (depth.clamp_min(1e-8)*math.tan(camera.FoVy*.5))
        previous = self.previous_levels[ids]
        current = schedule_levels(distance, screen, previous, self.distance_limits, self.screen_limits, self.hysteresis)
        near = depth-radius <= getattr(camera, 'znear', .01)
        current = torch.where(near, torch.zeros_like(current), current)
        changes = ((previous >= 0) & (previous != current)).sum()
        self.previous_levels[ids] = current
        data, selected, rows = self._assemble(visible, self.previous_levels)
        selected_owners = self.owners[selected]
        per_anchor = torch.bincount(selected_owners, minlength=self.anchor_count)
        anchor_counts = torch.bincount(current, minlength=len(self.labels))
        point_counts = torch.bincount(self.previous_levels[selected_owners], minlength=len(self.labels))
        nominal = self.level_ratios[self.previous_levels[self.owners[rows]]].sum()
        metrics = torch.cat((torch.stack((self.eligible[rows].sum(), changes, (per_anchor[ids] == 0).sum(), nominal)),
                             anchor_counts, point_counts)).cpu().tolist()
        stats = dict(total_gaussians=self.total, visible_anchors=len(ids), candidate_gaussians=len(rows),
            eligible_candidates=int(metrics[0]), active_gaussians=len(selected), level_changes=int(metrics[1]),
            empty_representation_anchors=int(metrics[2]), nominal_target_active=metrics[3],
            actual_fraction=len(selected)/max(1, self.total),
            budget_shortfall=max(0., metrics[3]-len(selected)))
        for j, label in enumerate(self.labels):
            stats['anchors_' + label] = int(metrics[4+j])
            stats['gaussians_' + label] = int(metrics[4+len(self.labels)+j])
        return data, selected, stats
