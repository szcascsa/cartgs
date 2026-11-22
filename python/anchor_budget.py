"""当前可见Anchor成员上的严格点数预算；ratio=B/N，选择与补偿使用同一条件。"""
import numbers
import math
from collections import OrderedDict
import time
import torch
from anchor_frustum import anchor_frustum_mask, compact_gaussians


def exact_budget_mask(scores, eligible, budget):
    """同分按原Gaussian行序优先；不使用分位数阈值，始终选择min(B,M)个点。"""
    if isinstance(budget, bool) or not isinstance(budget, numbers.Integral) or budget < 0:
        raise ValueError('Point budget must be a nonnegative integer')
    if scores.ndim != 1 or eligible.shape != scores.shape:
        raise ValueError('Scores and eligibility must align')
    rows = torch.nonzero(eligible).flatten()
    k = min(int(budget), len(rows))
    mask = torch.zeros_like(eligible, dtype=torch.bool)
    if k:
        order = scores[rows].argsort(descending=True, stable=True)
        mask[rows[order[:k]]] = True
    return mask


class FixedBudgetRenderer:
    def __init__(self, model, gaussians, config, seed=0, backend='pytorch'):
        from selector import AnchorSelectorNetwork
        if (not isinstance(model, AnchorSelectorNetwork) or not model.explicit_blocks or
                config.get('selection_mode') != 'flexgs' or config.get('entry_maturity', 0.) != 0 or
                config.get('protection_enabled', False) or
                getattr(model, 'reliability_gate_config', {}).get('enabled', False) or
                hasattr(model, 'gaussian_compensation') or not hasattr(model, 'spatial_compensation')):
            raise ValueError('Budget rendering requires Anchor FlexGS compensation without maturity/private residuals')
        if config.get('temperature', 1.) != 1.:
            raise ValueError('Budget rendering currently requires temperature=1')
        if not bool(model.spatial_compensation.normalization_ready):
            raise ValueError('Checkpoint compensation normalization is not initialized')
        if backend not in ('pytorch', 'cuda', 'cuda_graph'):
            raise ValueError('Inference backend must be pytorch, cuda or cuda_graph')
        self.backend = backend
        self.model, self.gaussians, self.seed = model, gaussians, int(seed)
        self.rebuild()

    def _signature(self):
        values = list(self.model.named_parameters()) + list(self.model.named_buffers())
        values += [(n, getattr(self.gaussians, n)) for n in
                   ('_xyz', '_scaling', '_rotation', '_opacity', '_features_dc', '_features_rest')]
        return (tuple((n, id(t), t.data_ptr(), t._version, tuple(t.shape)) for n, t in values),
                self.gaussians.active_sh_degree, tuple(getattr(self.model.spatial_compensation,'ratio_encoding_values',())))

    @torch.no_grad()
    def rebuild(self):
        anchors = self.model.anchors
        self.total = len(self.gaussians.get_xyz)
        if anchors.rows.shape != (self.total,):
            raise ValueError('Gaussian rows and Anchor owners must align')
        if (not torch.isfinite(anchors.centers).all() or not torch.isfinite(anchors.offset_scales).all() or
                not (anchors.offset_scales > 0).all()):
            raise ValueError('Invalid nominal Anchor cells')
        if self.total and (int(anchors.rows.min()) < 0 or int(anchors.rows.max()) >= len(anchors.centers)):
            raise ValueError('Gaussian Anchor owner outside the table')
        self.occupied = torch.bincount(anchors.rows, minlength=len(anchors.centers)) > 0
        generator = torch.Generator(device=self.gaussians.get_xyz.device).manual_seed(self.seed)
        self.noise = -self.gaussians.get_xyz.new_empty((self.total, 2)).exponential_(generator=generator).log()
        random_generator = torch.Generator(device=self.gaussians.get_xyz.device).manual_seed(self.seed ^ 0x5DEECE66D)
        self.random_scores = torch.rand(self.total, generator=random_generator, device=self.gaussians.get_xyz.device)
        self.knots = tuple(self.model.anchors.compensation_values)
        self.signature = self._signature()
        self.global_cache = OrderedDict()
        self.inference = None
        if self.backend != 'pytorch':
            from frozen_inference_cuda import FrozenResidualInference
            self.inference = FrozenResidualInference(self.model, self.gaussians, self.backend == 'cuda_graph')

    def check_state(self):
        if self.signature != self._signature() or tuple(self.model.anchors.compensation_values) != self.knots:
            raise RuntimeError('Budget renderer state changed; call rebuild() on the frozen model')

    @torch.no_grad()
    def prepare(self, camera, budget=None, *, ratio_budget=None, global_ratio=None,
                selection='learned', compensation=True, anchor_mask=None, view_conditioned=False,
                measure_time=True):
        from renderer import prepare_render_data
        from selector import slot_eligibility, compensated_attributes
        from selector_sampling import gumbel_selection, flexgs_inference_mask
        if sum(value is not None for value in (budget, ratio_budget, global_ratio)) != 1:
            raise ValueError('Use one of point budget, visible ratio budget, or global ratio')
        if not isinstance(view_conditioned,bool):raise ValueError('view_conditioned must be boolean')
        if view_conditioned and (camera is None or not (hasattr(self.model,'selector_view_residual') or hasattr(self.model,'selector_view_input_schema'))):
            raise ValueError('View-conditioned selection needs a camera-conditioned checkpoint and camera')
        if view_conditioned and self.inference is not None and not self.inference.joint_schema:
            raise ValueError('View residual currently requires the pytorch inference backend')
        if not isinstance(measure_time,bool):raise ValueError('measure_time must be boolean')
        if budget is not None and (isinstance(budget, bool) or not isinstance(budget, numbers.Integral) or budget < 0):
            raise ValueError('Point budget must be a nonnegative integer')
        for value in (ratio_budget, global_ratio):
            if value is not None and (isinstance(value, bool) or not isinstance(value, numbers.Real) or
                                      not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError('Ratio must be finite and in [0,1]')
        if global_ratio == 0:raise ValueError('Global training ratio must be positive')
        if selection not in ('learned', 'random') or not isinstance(compensation, bool):
            raise ValueError('Use learned/random selection and boolean compensation')
        started = time.perf_counter()
        self.check_state()
        anchors, pc = self.model.anchors, self.gaussians
        if anchor_mask is not None and (anchor_mask.shape != self.occupied.shape or anchor_mask.dtype != torch.bool or
                                        anchor_mask.device != self.occupied.device or global_ratio is not None):
            raise ValueError('Anchor mask must match the table; global ratio is filtered after selection')
        cache_key = (global_ratio, selection, compensation)
        if global_ratio is not None and not view_conditioned and cache_key in self.global_cache:
            data, selected, cached = self.global_cache[cache_key]
            self.global_cache.move_to_end(cache_key)
            return data, selected, dict(cached, selector_ms=0., compensation_ms=0., budget_calibration_ms=0., cache_hit=True,
                                       stage_timing_enabled=measure_time, prepare_ms=(time.perf_counter()-started)*1000)
        if global_ratio is not None:
            visible_count = None
            rows = torch.arange(self.total, device=pc.get_xyz.device)
            compact = pc
            available = None
        else:
            visible = (self.inference.frustum(camera.full_proj_transform, self.occupied)
                       if self.inference is not None and self.inference.dynamic_fusion else
                       anchor_frustum_mask(anchors.centers, anchors.offset_scales,
                                           camera.full_proj_transform, self.occupied))
            if anchor_mask is not None:visible = visible & anchor_mask
            if self.inference is None:
                visible_count = int(visible.sum())
                rows = torch.nonzero(visible[anchors.rows]).flatten()
                available = None
            else:rows, visible_count, available = self.inference.candidates(visible)
            compact = compact_gaussians(pc, rows) if self.inference is None else None
        eligible = (slot_eligibility(self.model, compact.get_opacity, rows) if self.inference is None
                    else self.inference.eligible[rows])
        n = len(rows)
        if available is None:available = int(eligible.sum())
        if global_ratio is not None:
            ratio = float(global_ratio) if n else None
            budget = n-int((1-float(global_ratio))*(n-1)) if n else 0
        elif ratio_budget is not None:
            ratio = float(ratio_budget) if n and ratio_budget else None
            budget = math.floor(float(ratio_budget)*n)
        else:
            ratio = min(1., int(budget)/n) if n and budget else None
        mask = torch.zeros(n, dtype=torch.bool, device=rows.device)
        native_ids = None
        attributes = None
        selector_ms = compensation_ms = calibration_ms = 0.
        if global_ratio is not None and selection == 'random':
            # 保留原分位数的同分行为：随机组以学习组的实际K作严格配对，首次缺少基准才校准。
            calibration_started = time.perf_counter()
            baseline = next((entry for key, entry in self.global_cache.items()
                             if key[:2] == (global_ratio, 'learned')), None)
            if baseline is None:
                _, reference_ids, _ = self.prepare(None, global_ratio=global_ratio,
                                                   selection='learned', compensation=False)
                budget = len(reference_ids)
            else:budget = len(baseline[1])
            calibration_ms = (time.perf_counter()-calibration_started)*1000
        if ratio is not None and available and (budget or global_ratio is not None):
            if measure_time and pc.get_xyz.is_cuda:torch.cuda.synchronize(pc.get_xyz.device)
            timer = time.perf_counter()
            shared = None
            anchor_code=((self.model.joint_view_code(camera) if self.inference is None else self.inference.view_code(camera))
                         if view_conditioned and hasattr(self.model,'joint_view_conditioning_schema') else None)
            if selection == 'random':
                if self.inference is None:mask = exact_budget_mask(self.random_scores[rows], eligible, budget)
                else:native_ids = self.inference.select(self.random_scores[rows], eligible, min(budget, available))
            else:
                if self.inference is None:
                    args=(compact.get_xyz,compact.get_scaling,compact.get_features_dc,compact.get_opacity,pc.get_xyz.new_tensor([[ratio]]))
                    if view_conditioned:
                        _,shared,logits=self.model.forward_view_selection_features(*args,camera,source_rows=rows,return_logits=True,anchor_code=anchor_code)
                    else:_,shared,logits=self.model.forward_selection_features(*args,source_rows=rows,return_logits=True)
                else:logits = self.inference.logits(rows, ratio, anchor_code, view_conditioned)
                if not torch.isfinite(logits).all():
                    raise RuntimeError('Nonfinite Selector logits at the requested ratio')
                if global_ratio is not None:
                    mask, _, _ = flexgs_inference_mask(logits, ratio, eligible=eligible, noise=self.noise)
                else:
                    scores = (self.inference.scores(logits, self.noise, rows)
                              if self.inference is not None and self.inference.dynamic_fusion else
                              gumbel_selection(logits, hard=False, noise=self.noise[rows]))
                    if self.inference is None:mask = exact_budget_mask(scores, eligible, budget)
                    else:native_ids = self.inference.select(scores, eligible, min(budget, available))
            if measure_time and pc.get_xyz.is_cuda:torch.cuda.synchronize(pc.get_xyz.device)
            selector_ms = (time.perf_counter()-timer)*1000 if measure_time else 0.
            if compensation and self.inference is None:
                timer = time.perf_counter()
                attributes = compensated_attributes(self.model, compact, mask, ratio, shared, source_rows=rows,
                    camera=camera if view_conditioned else None,anchor_code=anchor_code)
                if attributes is not None:
                    attributes['scaling'] = attributes['scaling'].log().exp()
                    attributes['rotation'] = pc.rotation_activation(attributes['rotation'])
                if measure_time and pc.get_xyz.is_cuda:torch.cuda.synchronize(pc.get_xyz.device)
                compensation_ms = (time.perf_counter()-timer)*1000 if measure_time else 0.
        selected = rows[mask] if native_ids is None else rows[native_ids]
        if self.inference is None:
            data = prepare_render_data(compact, mask, attributes)
            if any(not torch.isfinite(t).all() for t in data['inputs'].values() if isinstance(t, torch.Tensor)):
                raise RuntimeError('Nonfinite compensated attributes at the requested ratio')
        else:
            timer = time.perf_counter()
            data = self.inference.render_data(selected, ratio, compensation,
                anchor_code if ratio is not None and available and (budget or global_ratio is not None) else None,
                view_conditioned, capture=ratio_budget is None and (global_ratio is not None or len(selected)==int(budget)))
            if measure_time and pc.get_xyz.is_cuda:torch.cuda.synchronize(pc.get_xyz.device)
            compensation_ms = (time.perf_counter()-timer)*1000 if compensation and measure_time else 0.
        if (global_ratio is None or selection == 'random') and len(selected) != min(int(budget), available):
            raise RuntimeError('Strict point budget mismatch')
        stats = dict(total_gaussians=self.total, visible_anchors=visible_count,
            candidate_gaussians=n, eligible_candidates=available, requested_budget=int(budget),
            effective_budget=min(int(budget), available), active_gaussians=len(selected), ratio=ratio,
            selection_scope='global' if global_ratio is not None else 'visible_anchors',
            budget_unit='training_ratio' if global_ratio is not None else ('ratio' if ratio_budget is not None else 'points'),
            requested_ratio=global_ratio if global_ratio is not None else ratio_budget, cache_hit=False,
            inference_backend=self.backend,
            stage_timing_enabled=measure_time,
            selection=selection, compensation_enabled=compensation,
            ablation_group=('A' if compensation else 'B') if selection == 'learned' else ('C' if compensation else 'D'),
            budget_calibration_ms=calibration_ms,
            ratio_in_training_range=ratio is None or self.knots[0]-1e-8 <= ratio <= self.knots[-1]+1e-8,
            training_ratio_min=self.knots[0], training_ratio_max=self.knots[-1],
            budget_shortfall=max(0, int(budget)-len(selected)), selector_ms=selector_ms,
            compensation_ms=compensation_ms, prepare_ms=(time.perf_counter()-started)*1000)
        stats['view_conditioned']=view_conditioned and selection=='learned'
        if global_ratio is not None and not view_conditioned:
            self.global_cache[cache_key] = (data, selected, dict(stats))
            if len(self.global_cache) > 8:self.global_cache.popitem(last=False)
        return data, selected, stats
