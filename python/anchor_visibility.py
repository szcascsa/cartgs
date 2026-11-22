"""Viewer的Anchor 0/1/2层级显示范围；不修改模型或Slot状态。"""
import torch
from anchor_frustum import anchor_frustum_mask


class AnchorVisibility:
    @torch.no_grad()
    def __init__(self, anchors):
        self.owners = anchors.rows
        self.counts = torch.bincount(self.owners, minlength=len(anchors.centers))
        self.occupied = self.counts > 0
        self.levels = getattr(anchors, 'levels', torch.zeros_like(self.counts))
        if self.levels.shape != self.counts.shape or bool(((self.levels < 0) | (self.levels > 2)).any()):
            raise ValueError('Anchor levels must have one value in [0, 2] per Anchor')
        self.level_masks = torch.arange(3, device=self.owners.device)[:, None] == self.levels[None, :]
        occupied_levels = self.level_masks & self.occupied
        counts = torch.stack((occupied_levels.sum(1), (occupied_levels*self.counts).sum(1))).cpu().tolist()
        self.catalog = [dict(level=level, anchors=counts[0][level], gaussians=counts[1][level]) for level in range(3)]
        self.last_key, self.last_mask = None, None

    @torch.no_grad()
    def resolve(self, selected):
        if selected is None:selected = [0, 1, 2]
        if not isinstance(selected, list) or any(type(value) is not int or value not in (0, 1, 2) for value in selected):
            raise ValueError('anchor_levels must be a list of Anchor levels 0, 1, 2 or null')
        key = tuple(sorted(set(selected)))
        if key != self.last_key:
            mask = None if key == (0, 1, 2) else (self.level_masks[list(key)].any(0) if key else torch.zeros_like(self.occupied))
            self.last_key, self.last_mask = key, mask
        return self.last_mask

    @torch.no_grad()
    def frame_stats(self, anchors, camera, mask, ids, radii, inference=None):
        if inference is not None and getattr(inference, 'summary_fusion', False):
            # Recompute visibility for this call; no camera/ratio-result cache.
            visible = inference.frustum(camera.full_proj_transform, self.occupied)
            counts = inference.ext.frame_counts(visible, self.counts, self.levels,
                self.owners, ids, radii, inference.empty if mask is None else mask).cpu().tolist()
        else:
            visible = anchor_frustum_mask(anchors.centers, anchors.offset_scales, camera.full_proj_transform, self.occupied)
            enabled = self.occupied if mask is None else mask & self.occupied
            enabled_levels = self.level_masks & enabled
            visible_levels = self.level_masks & visible
            allowed_visible = visible_levels & enabled
            submitted_owners = self.owners[ids]
            projected_owners = submitted_owners[radii > 0]
            per_level = lambda rows: torch.bincount(self.levels[rows], minlength=3)
            counts = torch.stack((enabled_levels.sum(1), (enabled_levels*self.counts).sum(1),
                visible_levels.sum(1), (visible_levels*self.counts).sum(1),
                allowed_visible.sum(1), (allowed_visible*self.counts).sum(1),
                per_level(submitted_owners.unique()), per_level(projected_owners.unique()),
                per_level(submitted_owners), per_level(projected_owners))).cpu().tolist()
        levels = [dict(self.catalog[level], enabled=level in self.last_key,
            visible_anchors=counts[2][level], visible_gaussians=counts[3][level],
            submitted_gaussians=counts[8][level], projected_gaussians=counts[9][level]) for level in range(3)]
        return dict(anchor_filter_enabled=mask is not None,
            enabled_anchors=sum(counts[0]), enabled_anchor_gaussians=sum(counts[1]),
            view_anchors=sum(counts[2]), enabled_visible_anchors=sum(counts[4]),
            enabled_visible_gaussians=sum(counts[5]), submitted_anchors=sum(counts[6]),
            projected_anchors=sum(counts[7]), anchor_levels=levels)


@torch.no_grad()
def filter_render_packet(data, ids, anchor_mask, owners):
    """全图训练ratio保持原选择与补偿，再按Anchor层级隐藏；不改写缓存的原始Tensor。"""
    if anchor_mask is None:return data, ids
    keep = anchor_mask[owners[ids]]
    result = dict(data, inputs={name: value[keep].contiguous() if isinstance(value, torch.Tensor) else value
                               for name, value in data['inputs'].items()})
    return result, ids[keep]
