"""可见名义Anchor按投影尺寸/相机深度排序，按Gaussian权重分档并统一渲染。"""
import math
import numbers
import time
import torch
from anchor_frustum import anchor_frustum_mask, compact_gaussians
from anchor_budget import exact_budget_mask


DEFAULT_RATIOS = (.5, .3, .1)
DEFAULT_SHARES = (.2, .3, .5)
LEVEL_COLORS = ((1., .25, .15), (.15, .8, .25), (.15, .4, 1.))


def _triple(values, ratios=False):
    if (not isinstance(values, (tuple, list)) or len(values) != 3 or any(
            isinstance(x, bool) or not isinstance(x, numbers.Real) or not math.isfinite(x) for x in values)):
        raise ValueError('LoD needs three finite numeric ratios/shares')
    values = tuple(float(x) for x in values)
    if ratios:
        if not 1 >= values[0] > values[1] > values[2] > 0:
            raise ValueError('LoD ratios must satisfy 1 >= high > middle > low > 0')
    elif any(x < 0 or x > 1 for x in values) or not math.isclose(sum(values), 1., abs_tol=1e-6):
        raise ValueError('LoD shares must be nonnegative and sum to 1')
    return values


@torch.no_grad()
def anchor_projection(centers, scales, camera):
    """格子8角点；跨近裁剪面的格子以+inf尺寸保守排前，不裁剪屏幕边缘。"""
    signs = centers.new_tensor([[-1,-1,-1], [-1,-1,1], [-1,1,-1], [-1,1,1],
                               [1,-1,-1], [1,-1,1], [1,1,-1], [1,1,1]])
    corners = centers[:, None, :]+scales[:, None, :]*.5*signs
    homogeneous = torch.cat((corners, torch.ones_like(corners[:, :, :1])), 2)
    clip = homogeneous @ camera.full_proj_transform
    view = homogeneous @ camera.world_view_transform
    near = view[:, :, 2].amin(1) <= getattr(camera, 'znear', .01)
    ndc = clip[:, :, :2]/clip[:, :, 3:].clamp_min(torch.finfo(centers.dtype).tiny)
    extent = ndc.amax(1)-ndc.amin(1)
    pixel_extent = extent*centers.new_tensor([camera.image_width, camera.image_height])*.5
    screen = pixel_extent.amax(1)/camera.image_height
    screen = torch.where(near, torch.full_like(screen, float('inf')), screen)
    depth = (torch.cat((centers, torch.ones_like(centers[:, :1])), 1) @ camera.world_view_transform)[:, 2]
    return screen, depth, near


@torch.no_grad()
def rank_levels(metric, counts, uids, shares=DEFAULT_SHARES, *, descending=True, previous=None, hysteresis=0.):
    """整Anchor按累计Gaussian区间中点分组；目标占比允许一个Anchor粒度误差。"""
    shares = _triple(shares)
    if (metric.ndim != 1 or counts.shape != metric.shape or uids.shape != metric.shape or
            not 0 <= hysteresis < .25):
        raise ValueError('Invalid ranked Anchor layout or hysteresis')
    # 先以稳定UID排列，再稳定排序主指标，数组重排不改变同分优先级。
    order = uids.argsort(stable=True)
    order = order[metric[order].argsort(descending=descending, stable=True)]
    weight = counts[order].to(torch.float64)
    position = (weight.cumsum(0)-.5*weight)/weight.sum().clamp_min(1)
    limits = position.new_tensor([shares[0], shares[0]+shares[1]])
    cut = limits.expand(len(order), 2)
    if previous is not None and hysteresis:
        known = previous[order, None] >= 0
        coarse = previous[order, None] > torch.arange(2, device=order.device)
        cut = cut+torch.where(known, torch.where(coarse, -hysteresis, hysteresis), 0.)
    level = torch.empty_like(order)
    level[order] = (position[:, None] >= cut).sum(1)
    return level, order


class RankedAnchorRenderer:
    """复用预算Renderer的模型/随机键/可选CUDA；本类只保存有界分档历史。"""
    def __init__(self, budget_renderer):
        self.base = budget_renderer
        self.signature = None
        self.reset()

    def reset(self):
        self.previous = None
        self.history_key = None

    @torch.no_grad()
    def prepare(self, camera, *, sorting='projection', ratios=DEFAULT_RATIOS, shares=DEFAULT_SHARES,
                hysteresis=0., colorize=False, selection='learned', compensation=True,
                mixed_target=None, mixed_seed=0, mixed_sample=0, mixed_config=None, anchor_mask=None,view_conditioned=False):
        from selector import slot_eligibility, compensated_attributes
        from selector_sampling import gumbel_selection
        from renderer import prepare_render_data
        if mixed_target is None:
            ratios, shares = _triple(ratios, True), _triple(shares)
        if sorting not in ('projection', 'depth'):
            raise ValueError('LoD sorting must be projection or depth')
        if (isinstance(hysteresis, bool) or not isinstance(hysteresis, numbers.Real) or
                not math.isfinite(hysteresis) or not 0 <= hysteresis < .25):
            raise ValueError('LoD hysteresis must be finite in [0,.25)')
        if selection not in ('learned', 'random') or not isinstance(compensation, bool) or not isinstance(colorize, bool):
            raise ValueError('Invalid LoD selection/compensation/color controls')
        started = time.perf_counter()
        base = self.base;base.check_state()
        if not isinstance(view_conditioned,bool):raise ValueError('view_conditioned must be boolean')
        if view_conditioned and (not (hasattr(base.model,'selector_view_residual') or hasattr(base.model,'selector_view_input_schema')) or base.inference is not None):
            raise ValueError('View residual requires a view-residual checkpoint and pytorch backend')
        distribution = None
        if mixed_target is not None:
            from mixed_ratio import draw, calibrate, integer_budgets
            for endpoint in (base.knots[0], base.knots[-1]):
                if abs(mixed_target-endpoint) <= 1e-7:mixed_target = endpoint
            if sorting != 'projection' or hysteresis:
                raise ValueError('Gaussian mixed ratio uses longest-side sorting without hysteresis')
            distribution = draw(mixed_target, base.knots[0], base.knots[-1], mixed_seed, mixed_sample,
                                **(mixed_config or {}))
            shares = distribution['shares'] if distribution['sigma'] > 0 else (1., 0., 0.)
        pc, model = base.gaussians, base.model
        anchors, engine = model.anchors, base.inference
        if anchor_mask is not None and (anchor_mask.shape != base.occupied.shape or anchor_mask.dtype != torch.bool or
                                        anchor_mask.device != base.occupied.device):
            raise ValueError('Anchor visibility mask must match the Anchor table')
        if self.signature != base.signature:
            self.reset();self.signature = base.signature
            self.counts = torch.bincount(anchors.rows, minlength=len(anchors.centers))
            self.uids = getattr(anchors, 'block_uids', torch.arange(len(anchors.centers), device=anchors.rows.device))
            if self.uids.shape != (len(anchors.centers),) or len(self.uids.unique()) != len(self.uids):
                raise ValueError('LoD requires unique stable Anchor UIDs')
        history_key = (sorting, tuple(ratios), tuple(shares), float(hysteresis), mixed_target, mixed_seed, mixed_sample)
        if self.history_key != history_key:
            self.previous = torch.full_like(self.counts, -1);self.history_key = history_key
        visible = anchor_frustum_mask(anchors.centers, anchors.offset_scales, camera.full_proj_transform, base.occupied)
        if anchor_mask is not None:visible = visible & anchor_mask
        visible_ids = torch.nonzero(visible).flatten()
        screen, depth, near = anchor_projection(anchors.centers[visible_ids], anchors.offset_scales[visible_ids], camera)
        levels, _ = rank_levels(screen if sorting == 'projection' else depth,
            self.counts[visible_ids], self.uids[visible_ids], shares, descending=sorting == 'projection',
            previous=self.previous[visible_ids], hysteresis=hysteresis)
        self.previous.fill_(-1);self.previous[visible_ids] = levels
        anchor_levels = torch.full_like(self.counts, -1);anchor_levels[visible_ids] = levels
        group_counts = [int(self.counts[visible_ids][levels == i].sum()) for i in range(3)]
        budgets = None
        if distribution is not None:
            ratios = calibrate(distribution, group_counts, mixed_target, base.knots[0], base.knots[-1])
            budgets = integer_budgets(group_counts, ratios, mixed_target)
        if pc.get_xyz.is_cuda:torch.cuda.synchronize(pc.get_xyz.device)
        grouping_ms = (time.perf_counter()-started)*1000
        packets, selected_ids, selected_levels, groups = [], [], [], []
        selector_ms = compensation_ms = 0.
        for level, ratio in enumerate(ratios):
            rows = torch.nonzero(anchor_levels[anchors.rows] == level).flatten()
            compact = compact_gaussians(pc, rows) if engine is None else None
            eligible = (slot_eligibility(model, compact.get_opacity, rows) if engine is None else engine.eligible[rows])
            n, available = len(rows), int(eligible.sum())
            requested = math.floor(ratio*n) if budgets is None else budgets[level]
            k = min(requested, available)
            tick = time.perf_counter()
            shared = None
            if k:
                if selection == 'random':scores = base.random_scores[rows]
                else:
                    if engine is None:
                        args=(compact.get_xyz,compact.get_scaling,compact.get_features_dc,compact.get_opacity,pc.get_xyz.new_tensor([[ratio]]))
                        if view_conditioned:
                            _,shared,logits=model.forward_view_selection_features(*args,camera,source_rows=rows,return_logits=True)
                        else:_,shared,logits=model.forward_selection_features(*args,source_rows=rows,return_logits=True)
                    else:logits = engine.logits(rows, ratio)
                    if not torch.isfinite(logits).all():raise RuntimeError('Nonfinite ranked LoD Selector logits')
                    scores = gumbel_selection(logits, hard=False, noise=base.noise[rows])
                if engine is None:
                    mask = exact_budget_mask(scores, eligible, requested)
                    local_ids = torch.nonzero(mask).flatten()
                else:local_ids = engine.select(scores, eligible, k)
                ids = rows[local_ids]
            else:
                ids = rows[:0]
                mask = torch.zeros(n, dtype=torch.bool, device=rows.device)
            if pc.get_xyz.is_cuda:torch.cuda.synchronize(pc.get_xyz.device)
            selector_ms += (time.perf_counter()-tick)*1000
            tick = time.perf_counter()
            if engine is None:
                attributes = compensated_attributes(model, compact, mask, ratio, shared, source_rows=rows) if compensation else None
                if attributes is not None:
                    attributes['scaling'] = attributes['scaling'].log().exp()
                    attributes['rotation'] = pc.rotation_activation(attributes['rotation'])
                packet = prepare_render_data(compact, mask, attributes)
            else:packet = engine.render_data(ids, ratio, compensation)
            if pc.get_xyz.is_cuda:torch.cuda.synchronize(pc.get_xyz.device)
            compensation_ms += (time.perf_counter()-tick)*1000
            if len(ids) != k:raise RuntimeError('Ranked LoD strict K mismatch')
            packets.append(packet);selected_ids.append(ids);selected_levels.append(torch.full_like(ids, level))
            groups.append(dict(level=level, ratio=ratio, target_share=shares[level], actual_share=n/max(1,sum(group_counts)),
                anchors=int((levels == level).sum()), candidate_gaussians=n, eligible_candidates=available,
                requested_budget=requested, active_gaussians=k, budget_shortfall=requested-k))
        # 三档属性恢复原全局行序后只调用一次Renderer，禁止逐档合成图像。
        ids = torch.cat(selected_ids);order = ids.argsort(stable=True);ids = ids[order]
        point_levels = torch.cat(selected_levels)[order]
        inputs = {}
        for name, value in packets[0]['inputs'].items():
            inputs[name] = None if value is None else torch.cat([p['inputs'][name] for p in packets], 0)[order].contiguous()
        if any(not torch.isfinite(value).all() for value in inputs.values() if value is not None):
            raise RuntimeError('Nonfinite ranked LoD render attributes')
        data = dict(sh_degree=pc.active_sh_degree, inputs=inputs)
        if colorize:data['lod_colors'] = pc.get_xyz.new_tensor(LEVEL_COLORS)[point_levels].contiguous()
        n = sum(group['candidate_gaussians'] for group in groups)
        requested = sum(group['requested_budget'] for group in groups)
        stats = dict(total_gaussians=base.total, visible_anchors=len(visible_ids), candidate_gaussians=n,
            eligible_candidates=sum(g['eligible_candidates'] for g in groups), requested_budget=requested,
            effective_budget=sum(g['active_gaussians'] for g in groups), active_gaussians=len(ids), ratio=None,
            actual_retention_ratio=len(ids)/n if n else 0., weighted_target_ratio=sum(g['ratio']*g['candidate_gaussians'] for g in groups)/n if n else 0.,
            requested_ratio=None, selection_scope='ranked_visible_anchors', budget_unit='ranked_lod',
            cache_hit=False, sorting=sorting, lod_ratios=list(ratios), lod_shares=list(shares), lod_groups=groups,
            lod_hysteresis=float(hysteresis), lod_colorize=colorize, near_plane_anchors=int(near.sum()),
            selection=selection, compensation_enabled=compensation,
            ablation_group=('A' if compensation else 'B') if selection == 'learned' else ('C' if compensation else 'D'),
            inference_backend=base.backend, budget_calibration_ms=0., selector_ms=selector_ms,
            compensation_ms=compensation_ms, grouping_ms=grouping_ms, prepare_ms=(time.perf_counter()-started)*1000,
            budget_shortfall=requested-len(ids),
            ratio_in_training_range=all(base.knots[0]-1e-8 <= r <= base.knots[-1]+1e-8 for r,g in zip(ratios,groups) if g['candidate_gaussians']))
        stats.update(mixed_ratio=distribution is not None, target_equivalent_ratio=mixed_target,
            view_conditioned=view_conditioned and selection=='learned',
            mixed_seed=mixed_seed if distribution is not None else None,
            mixed_sample=mixed_sample if distribution is not None else None,
            mixed_sigma=distribution['sigma'] if distribution is not None else None)
        return data, ids, stats
