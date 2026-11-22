"""按Anchor名义格子筛选离线候选，保持全图FlexGS分位数预算。"""
import math
import copy
import torch
from selector_sampling import gumbel_selection


@torch.no_grad()
def anchor_frustum_mask(centers, offset_scales, full_projection, occupied=None):
    """行向量相机矩阵；六平面保守AABB测试，边界相交也保留。"""
    if centers.ndim != 2 or centers.shape[1] != 3 or offset_scales.shape != (len(centers), 1):
        raise ValueError('Anchor bounds require Ax3 centers and Ax1 cell side lengths')
    if full_projection.shape != (4, 4):
        raise ValueError('Expected the renderer full world-to-clip projection')
    m = full_projection
    planes = torch.stack((m[:, 3]+m[:, 0], m[:, 3]-m[:, 0],
                          m[:, 3]+m[:, 1], m[:, 3]-m[:, 1],
                          m[:, 2], m[:, 3]-m[:, 2]))
    distance = centers @ planes[:, :3].T + planes[:, 3]
    radius = (offset_scales * .5) * planes[:, :3].abs().sum(1)
    visible = (distance + radius >= 0).all(1)
    return visible if occupied is None else visible & occupied


@torch.no_grad()
def global_budget_mask(logits, eligible, ratio, global_count, noise):
    """等价于把非候选补成-inf后，沿用原全图排序分位数；不是新增Top-K。"""
    if not math.isfinite(ratio) or not 0 < ratio <= 1 or global_count < len(logits):
        raise ValueError('Invalid ratio or global population')
    if eligible.shape != (len(logits),) or noise.shape != logits.shape:
        raise ValueError('Compact logits, eligibility and fixed noise must align')
    if not len(logits):
        return eligible.bool(), None
    scores = gumbel_selection(logits, hard=False, noise=noise)
    scores = scores.masked_fill(~eligible, -float('inf'))
    index = int((1-ratio) * (global_count-1))
    local_index = index - (global_count-len(logits))
    threshold = scores.new_full((), -float('inf')) if local_index < 0 else scores.sort().values[local_index]
    return eligible & (scores >= threshold), threshold


def compact_gaussians(gaussians, rows):
    result = copy.copy(gaussians)
    for name in ['_xyz', '_scaling', '_rotation', '_features_dc', '_features_rest', '_opacity']:
        setattr(result, name, getattr(gaussians, name).index_select(0, rows))
    return result


class AnchorFrustumRenderer:
    """冻结Master上逐相机查询；随机键每ratio固定一次，所有视角按原全局行索引复用。"""
    @torch.no_grad()
    def __init__(self, model, gaussians, ratio, config, temperature=1., protection_enabled=False):
        from selector import AnchorSelectorNetwork, mask_inactive_slots
        if (not isinstance(model, AnchorSelectorNetwork) or not model.explicit_blocks or
                config.get('selection_mode') != 'flexgs' or config.get('entry_maturity', 0.) != 0 or
                protection_enabled or hasattr(model, 'gaussian_compensation') or
                not hasattr(model, 'spatial_compensation')):
            raise ValueError('Anchor-frustum mode requires current FlexGS spatial compensation, entry=0, no legacy protection')
        if temperature != 1.:
            raise ValueError('Anchor-frustum evaluation currently requires the baseline temperature=1')
        anchors = model.anchors
        if (not torch.isfinite(anchors.centers).all() or not torch.isfinite(anchors.offset_scales).all() or
                not (anchors.offset_scales > 0).all()):
            raise ValueError('Non-finite or non-positive Anchor nominal bounds')
        self.model, self.gaussians, self.ratio = model, gaussians, float(ratio)
        self.temperature = temperature
        self.total = len(gaussians.get_xyz)
        if anchors.rows.numel() != self.total:
            raise ValueError('Master rows and Anchor owners must align')
        self.occupied = torch.bincount(anchors.rows, minlength=len(anchors.centers)) > 0
        self.eligible = mask_inactive_slots(model, torch.ones(self.total, device=gaussians.get_xyz.device),
                                             gaussians.get_opacity).bool()
        # 与旧eval的采样调用相同；只生成一次，逐视角不改变同一Gaussian的随机键。
        self.noise = -gaussians.get_xyz.new_empty((self.total, 2)).exponential_().log()

    @torch.no_grad()
    def prepare(self, camera):
        from selector import compensated_attributes
        from renderer import prepare_render_data
        anchors = self.model.anchors
        visible = anchor_frustum_mask(anchors.centers, anchors.offset_scales,
                                      camera.full_proj_transform, self.occupied)
        rows = torch.nonzero(visible[anchors.rows]).flatten()
        compact = compact_gaussians(self.gaussians, rows)
        if len(rows):
            _, shared, logits = self.model.forward_selection_features(
                compact.get_xyz, compact.get_scaling, compact.get_features_dc, compact.get_opacity,
                compact.get_xyz.new_tensor([[self.ratio]]), self.temperature,
                source_rows=rows, return_logits=True)
            mask, threshold = global_budget_mask(logits, self.eligible[rows], self.ratio,
                                                   self.total, self.noise[rows])
            attributes = compensated_attributes(self.model, compact, mask, self.ratio, shared,
                                                source_rows=rows)
            if attributes is not None:
                # 原eval会先导出PLY再重载；在GPU上保留同样的scale/rotation变换，不写逐帧PLY。
                attributes['scaling'] = attributes['scaling'].log().exp()
                attributes['rotation'] = compact.rotation_activation(attributes['rotation'])
        else:
            mask = torch.zeros(0, device=rows.device, dtype=torch.bool)
            attributes, threshold = None, None
        selected = rows[mask]
        render_data = prepare_render_data(compact, mask, attributes)
        target = self.ratio * self.total
        eligible_count = int(self.eligible[rows].sum())
        return render_data, selected, dict(
            total_gaussians=self.total, visible_anchors=int(visible.sum()),
            candidate_gaussians=len(rows), eligible_candidates=eligible_count,
            active_gaussians=len(selected), target_active=target,
            legacy_quantile_budget=self.total-int((1-self.ratio)*(self.total-1)) if self.total else 0,
            budget_shortfall=max(0., target-len(selected)),
            selection_threshold=float(threshold) if threshold is not None and torch.isfinite(threshold) else None)
