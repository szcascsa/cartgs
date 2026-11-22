"""Selector inference helpers shared by the LOD evaluation script."""

import os

import numpy as np
import torch
from torch import nn
from plyfile import PlyData
from elastic_budget import classify_protection, protection_statistics
from elastic_entry import elastic_selector_probability


class SelectorNetwork(nn.Module):
    """Python counterpart of improvements::selector::GumbelNetwork."""

    embedding_width = 32

    def __init__(self):
        super().__init__()
        # The evaluation config selects cuda by default; keep bare CPU/oracle modules legacy.
        self.selector_backend = "legacy"
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
        if self.selector_backend == "cuda":
            from selector_cuda_backend import network
            logits = network((position, rotation, scale, target_ratio), self.parameters())
            return torch.softmax(logits / tau, dim=1)[:, 1]
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

    def set_backend(self, backend):
        if backend not in ("legacy", "cuda"):
            raise ValueError("Selector.backend must be legacy or cuda")
        self.selector_backend = backend


class AnchorState(nn.Module):
    def __init__(self, anchors, population, ratio_knots=None, feature_dim=16, ratio_shared_dim=8):
        super().__init__()
        if ratio_knots is not None and ratio_shared_dim not in (8,16):
            raise ValueError('Ratio shared Feature width must be 8 or 16')
        width=ratio_shared_dim if ratio_knots is not None else feature_dim
        self.features = nn.Parameter(torch.zeros(anchors,width),requires_grad=width>0)
        if ratio_knots is not None:
            knots = torch.as_tensor(ratio_knots, dtype=torch.float64).cpu()
            if (knots.ndim != 1 or len(knots)<2 or not torch.isfinite(knots).all() or
                    (knots<=0).any() or (knots>1).any() or not (knots[1:]>knots[:-1]).all()):
                raise ValueError("Ratio feature knots must be strictly increasing in (0,1]")
            self.register_buffer("ratio_knots", knots.clone())
            self.ratio_values = knots.tolist()
            for i in range(len(knots)):
                self.register_parameter("ratio_features_"+str(i), nn.Parameter(torch.zeros(anchors,8)))
        self.register_buffer("centers", torch.zeros(anchors, 3))
        self.register_buffer("keys", torch.zeros(anchors, 3, dtype=torch.long))
        self.register_buffer("rows", torch.zeros(population, dtype=torch.long))
        self.register_buffer("voxel_size", torch.tensor(0.1, dtype=torch.float64))

    def lookup(self, ids, ratio):
        shared = self.features[ids]
        if not hasattr(self,"ratio_knots"):
            return shared
        value = float(ratio.reshape(-1)[0])
        if not np.isfinite(value) or not 0<value<=1 or (ratio.numel()>1 and not (ratio==value).all()):
            raise ValueError("Ratio-feature forward requires one finite ratio in (0,1]")
        hi = 0
        while hi+1<len(self.ratio_values) and value>self.ratio_values[hi]+1e-8:
            hi += 1
        current = getattr(self,"ratio_features_"+str(hi))[ids]
        if hi and abs(value-self.ratio_values[hi])>1e-8 and value<self.ratio_values[-1]:
            t = (value-self.ratio_values[hi-1])/(self.ratio_values[hi]-self.ratio_values[hi-1])
            current = (1-t)*getattr(self,"ratio_features_"+str(hi-1))[ids]+t*current
        return torch.cat((shared,current),dim=1)

    def compensation_lookup(self, source_rows, ratio):
        value = float(ratio)
        if not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError('Invalid compensation feature ratio')
        ids = self.rows[source_rows]
        hi = 0
        while hi+1 < len(self.compensation_values) and value > self.compensation_values[hi]+1e-8:
            hi += 1
        current = getattr(self, 'compensation_features_'+str(hi))[ids]
        if hi and abs(value-self.compensation_values[hi]) > 1e-8 and value < self.compensation_values[-1]:
            t = (value-self.compensation_values[hi-1])/(self.compensation_values[hi]-self.compensation_values[hi-1])
            current = (1-t)*getattr(self, 'compensation_features_'+str(hi-1))[ids]+t*current
        return current


class GaussianCompensationBank(nn.Module):
    """Checkpoint-compatible Gaussian identity residuals; no Selector inputs."""
    def __init__(self, population, independent, knots):
        super().__init__()
        values = knots.detach().cpu().double().flatten().tolist()
        if not values or any(not np.isfinite(x) or x <= 0 or x > 1 for x in values) or any(a >= b for a, b in zip(values, values[1:])):
            raise ValueError('Invalid private compensation ratio knots')
        self.independent = independent
        self.values = values
        self.register_buffer('ratio_knots', knots.clone())
        self.register_buffer('gaussian_uids', torch.zeros(population, dtype=torch.long))
        self.register_buffer('next_uid', torch.zeros((), dtype=torch.long))
        for i in range(len(values) if independent else 1):
            self.register_parameter('residual_'+str(i), nn.Parameter(torch.zeros(population, 14)))

    def forward(self, source_rows, ratio):
        ratio = float(ratio)
        if not np.isfinite(ratio) or not 0 < ratio <= 1:
            raise ValueError('Invalid private compensation ratio')
        if not self.independent:
            return self.residual_0[source_rows]
        hi = 0
        while hi + 1 < len(self.values) and ratio > self.values[hi] + 1e-8:
            hi += 1
        upper = getattr(self, 'residual_'+str(hi))[source_rows]
        if hi == 0 or ratio >= self.values[-1] or abs(ratio-self.values[hi]) <= 1e-8:
            return upper
        t = (ratio-self.values[hi-1])/(self.values[hi]-self.values[hi-1])
        return (1-t)*getattr(self, 'residual_'+str(hi-1))[source_rows] + t*upper


class AnchorSelectorNetwork(nn.Module):
    """Same feature/derived-offset/ratio network as the native Anchor path."""
    def __init__(self, anchors, population, joint_conditioning=True, gaussian_attributes=False, global_xyz=False, explicit_blocks=False, ratio_knots=None, feature_dim=16, ratio_scalar_input=0, ratio_shared_dim=8):
        super().__init__()
        self.selector_backend = "legacy"
        if ratio_knots is not None and not (explicit_blocks and gaussian_attributes and joint_conditioning and not global_xyz):
            raise ValueError("Ratio features require explicit attribute-conditioned Anchors")
        if feature_dim!=16 and not (feature_dim in (0,32) and ratio_knots is None and explicit_blocks and gaussian_attributes):
            raise ValueError("0D/32D Anchor features require the non-ratio explicit attribute path")
        if feature_dim==0 and (global_xyz or not joint_conditioning):
            raise ValueError('S00R00 requires the joint attribute/log-ratio Encoder')
        self.anchors = AnchorState(anchors, population, ratio_knots,feature_dim,ratio_shared_dim)
        self.ratio_features = ratio_knots is not None
        self.ratio_scalar_input = int(ratio_scalar_input)
        if self.ratio_scalar_input not in (0,1,2) or (self.ratio_scalar_input and not self.ratio_features):
            raise ValueError('Ratio scalar input requires ratio features and mode0/1/2')
        if self.ratio_scalar_input:
            self.register_buffer('anchor_ratio_scalar_input',torch.tensor(self.ratio_scalar_input,dtype=torch.long))
        self.joint_conditioning = joint_conditioning
        self.gaussian_attributes = gaussian_attributes
        self.global_xyz = global_xyz
        self.explicit_blocks = explicit_blocks
        if global_xyz or explicit_blocks:
            self.anchors.register_buffer("offset_scales", torch.zeros(anchors, 1))
        if explicit_blocks:
            self.anchors.register_buffer("explicit_blocks", torch.tensor(1, dtype=torch.long))
            self.anchors.register_buffer("block_uids", torch.arange(anchors))
            self.anchors.register_buffer("next_uid", torch.tensor(anchors, dtype=torch.long))
        self.local_emd = SelectorNetwork._make_embedding(11 if feature_dim==0 else (43 if feature_dim==32 else (ratio_shared_dim+18+int(bool(self.ratio_scalar_input)) if self.ratio_features else (30 if global_xyz else (27 if gaussian_attributes else (20 if joint_conditioning else 19))))))
        if joint_conditioning:
            self.register_buffer("anchor_layout_version", torch.tensor(10 if feature_dim==0 else (8 if feature_dim==32 else ((9 if ratio_shared_dim==16 else 7) if self.ratio_features else (5 if explicit_blocks else (4 if global_xyz else (3 if gaussian_attributes else 2))))), dtype=torch.long))
        else:
            self.time_emd = SelectorNetwork._make_embedding(1)
        self.soft_net = nn.Linear(32 if joint_conditioning else 64, 2)
        if feature_dim==0:
            nn.init.zeros_(self.soft_net.weight);nn.init.zeros_(self.soft_net.bias)

    def set_backend(self, backend):
        if backend != "legacy":
            raise ValueError("Anchor Selector uses LibTorch; old CUDA kernel has a different layout")
        self.selector_backend = backend

    def forward(self, position, rotation, scale, target_ratio, tau=1.0, source_rows=None,
                sh_dc=None, opacity=None):
        if hasattr(self,"compensation") or hasattr(self,'spatial_compensation'):
            return self.forward_selection_features(position,scale,sh_dc,opacity,target_ratio,tau,source_rows)[0]
        ids = self.anchors.rows if source_rows is None else self.anchors.rows[source_rows]
        if position.shape[0] != ids.numel():
            raise ValueError("Anchor Selector compact input requires source row indices")
        reference_scale = self.anchors.offset_scales[ids] if (self.global_xyz or self.explicit_blocks) else self.anchors.voxel_size
        offset = (position - self.anchors.centers[ids]) / reference_scale
        if self.joint_conditioning:
            ratio = target_ratio.log().expand(position.shape[0], 1)
            inputs = [self.anchors.lookup(ids,target_ratio), offset]
            if self.global_xyz:
                inputs += [position]
            if self.gaussian_attributes:
                if scale is None or sh_dc is None or opacity is None:
                    raise ValueError("Anchor attribute inputs require activated scale, SH DC and activated opacity")
                dc = sh_dc.squeeze(1) if sh_dc.ndim == 3 else sh_dc
                if scale.shape != position.shape or dc.shape != position.shape or opacity.shape != (position.shape[0], 1):
                    raise ValueError("Anchor attribute rows/shapes must match XYZ")
                inputs += [scale, dc, opacity]
            if self.ratio_features and self.ratio_scalar_input:
                inputs += [(target_ratio.log() if self.ratio_scalar_input==1 else target_ratio).expand(position.shape[0],1)]
            local = self.local_emd(torch.cat(inputs if self.ratio_features else inputs+[ratio], dim=1))
            return torch.softmax(self.soft_net(local)/tau, dim=1)[:, 1]
        local = self.local_emd(torch.cat((self.anchors.features[ids], offset), dim=1))
        ratio = self.time_emd(target_ratio)
        if ratio.shape[0] == 1:
            ratio = ratio.expand(position.shape[0], -1)
        return torch.softmax(self.soft_net(torch.cat((local, ratio), dim=1))/tau, dim=1)[:, 1]

    def forward_selection_features(self, position, scale, dc, opacity, ratio, tau=1., source_rows=None):
        ids=self.anchors.rows if source_rows is None else self.anchors.rows[source_rows]
        offset=(position-self.anchors.centers[ids])/self.anchors.offset_scales[ids]
        dc=dc.squeeze(1) if dc.ndim==3 else dc
        inputs=[self.anchors.lookup(ids,ratio),offset,scale,dc,opacity]
        if not self.ratio_features:
            inputs += [ratio.log().expand(len(position),1)]
        elif self.ratio_scalar_input:
            inputs += [(ratio.log() if self.ratio_scalar_input==1 else ratio).expand(len(position),1)]
        value=torch.cat(inputs,1)
        shared=self.local_emd[1](self.local_emd[0](value))
        hidden=self.local_emd[3](self.local_emd[2](shared))
        return torch.softmax(self.soft_net(hidden)/tau,dim=1)[:,1],shared


def load_selector(path, device):
    """Load a selector checkpoint produced by the C++ or Python path."""
    if not os.path.isfile(path):
        raise FileNotFoundError("selector checkpoint not found: {}".format(path))

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
    if "anchors.features" in state:
        blocks = "anchors.explicit_blocks" in state
        width = state["local_emd.0.weight"].shape[1]
        if width not in (11, 19, 20, 26, 27, 30, 34, 35, 43):
            raise RuntimeError("Unknown Anchor Selector input layout: "+str(width))
        has_ratio = 'anchors.ratio_knots' in state
        scalar_input = int(state.get('anchor_ratio_scalar_input',0))
        shared_width=state['anchors.features'].shape[1]
        if has_ratio and (width != shared_width+18+int(bool(scalar_input)) or shared_width not in (8,16)):
            raise RuntimeError('Ratio feature/scalar input layout mismatch')
        model = AnchorSelectorNetwork(state["anchors.features"].shape[0], state["anchors.rows"].numel(),
                                      joint_conditioning=width != 19, gaussian_attributes=width in (11, 26, 27, 30, 34, 35, 43), global_xyz=width == 30, explicit_blocks=blocks,
                                      ratio_knots=state.get("anchors.ratio_knots") if has_ratio else None,feature_dim=0 if width==11 else (32 if width==43 else 16),
                                      ratio_scalar_input=scalar_input,ratio_shared_dim=shared_width if has_ratio else 8).to(device)
        if width==26 and ("anchors.ratio_knots" not in state or state["anchors.features"].shape[1]!=8):
            raise RuntimeError("Ratio feature checkpoint metadata is missing or inconsistent")
        if "compensation_schema" in state:
            schema=int(state['compensation_schema'])
            if not blocks or width not in (11,26,27,34,35,43) or schema not in (1,2):
                raise RuntimeError('Unsupported Selector compensation layout')
            anchor_schema=int(state.get('anchor_ratio_compensation_schema',0))
            if 'spatial_compensation_schema' in state:
                if int(state['spatial_compensation_schema'])!=1 or schema!=2 or anchor_schema or 'private_compensation_mode' in state:
                    raise RuntimeError('Unsupported Spatial compensation checkpoint')
                from spatial_compensation import SpatialCompensation
                model.spatial_compensation=SpatialCompensation().to(device)
                model.register_buffer('spatial_compensation_schema',torch.zeros_like(state['spatial_compensation_schema']))
            elif anchor_schema in (2,3):
                model.compensation=nn.Sequential(nn.Linear(82 if anchor_schema==3 else 40,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,14)).to(device)
            else:
                model.compensation=nn.Sequential(nn.Linear(46 if anchor_schema else 32,32),nn.ReLU(),nn.Linear(32,10 if schema==1 else 14)).to(device)
            model.register_buffer('compensation_schema',torch.zeros_like(state['compensation_schema']))
        if 'anchor_ratio_compensation_schema' in state:
            anchor_schema=int(state['anchor_ratio_compensation_schema'])
            if (anchor_schema not in (1,2,3) or (width not in (26,27,34,35) if anchor_schema==1 else width!=11)
                    or 'private_compensation_mode' in state or int(state.get('compensation_schema',0))!=2):
                raise RuntimeError('Unsupported Anchor ratio compensation layout')
            extra_width=50 if anchor_schema==3 else (8 if anchor_schema==2 else 14)
            knots = state['anchors.compensation_ratio_knots']
            model.anchors.compensation_values = knots.detach().cpu().double().tolist()
            if len(model.anchors.compensation_values) < 2 or any(not 0 < x <= 1 for x in model.anchors.compensation_values) or any(
                    a >= b for a, b in zip(model.anchors.compensation_values, model.anchors.compensation_values[1:])):
                raise RuntimeError('Invalid Anchor compensation knots')
            if has_ratio and model.anchors.compensation_values != model.anchors.ratio_values:
                raise RuntimeError('Selector and compensation ratio knots must match')
            model.anchors.register_buffer('compensation_ratio_knots', torch.zeros_like(knots))
            for j in range(len(knots)):
                table = state['anchors.compensation_features_'+str(j)]
                if table.shape != (state['anchors.features'].shape[0],extra_width):
                    raise RuntimeError('Anchor compensation feature shape mismatch')
                model.anchors.register_parameter('compensation_features_'+str(j), nn.Parameter(torch.zeros_like(table)))
            model.register_buffer('anchor_ratio_compensation_schema', torch.zeros_like(state['anchor_ratio_compensation_schema']))
        if 'private_compensation_mode' in state:
            mode = int(state['private_compensation_mode'])
            if mode not in (2, 3) or int(state.get('compensation_schema', 0)) != 2:
                raise RuntimeError('Unsupported Gaussian-private compensation schema')
            population = state['anchors.rows'].numel()
            model.gaussian_compensation = GaussianCompensationBank(population, mode == 3,
                state['gaussian_compensation.ratio_knots']).to(device)
            uids = state['gaussian_compensation.gaussian_uids']
            if uids.dtype != torch.long or uids.shape != (population,) or (uids < 0).any() or len(torch.unique(uids)) != population:
                raise RuntimeError('Gaussian-private identities do not match population')
            model.register_buffer('private_compensation_mode', torch.zeros_like(state['private_compensation_mode']))
            for parameter in model.compensation.parameters():
                parameter.requires_grad_(False)
        if "anchors.levels" in state:
            model.anchors.register_buffer("levels", torch.zeros_like(state["anchors.levels"]))
            model.anchors.register_buffer("branches", torch.zeros_like(state["anchors.branches"]))
        variable = "anchors.member_counts" in state
        if variable:
            model.anchors.register_buffer("member_counts", torch.zeros_like(state["anchors.member_counts"]))
            model.anchors.register_buffer("topology_round", torch.zeros_like(state["anchors.topology_round"]))
        if "anchors.slot_inactive" in state:
            if not blocks:
                raise RuntimeError("Slot deactivation requires explicit ten-slot anchors")
            for name in ("slot_inactive", "slot_recovered"):
                value = state["anchors."+name]
                shape=(state["anchors.rows"].numel(),) if variable else (state["anchors.features"].shape[0],10)
                if value.dtype != torch.bool or value.shape != shape:
                    raise RuntimeError("Invalid slot deactivation checkpoint layout")
                model.anchors.register_buffer(name, torch.zeros_like(value))
        if width != 19 and int(state["anchor_layout_version"]) != (10 if width==11 else (8 if width==43 else ((9 if shared_width==16 else 7) if has_ratio else (5 if blocks else (4 if width == 30 else (3 if width == 27 else 2)))))):
            raise RuntimeError("Anchor Selector checkpoint layout version mismatch")
        if width==11 and (shared_width!=0 or has_ratio or (int(state.get('anchor_ratio_compensation_schema',0)) not in (2,3) and 'spatial_compensation_schema' not in state)):
            raise RuntimeError('S00R00 requires zero Selector latent and a latent8/50 deep compensation head')
        if blocks:
            anchors = state["anchors.features"].shape[0]
            if variable:
                owners=state["anchors.rows"]
                counts=state["anchors.member_counts"]
                if (width not in (11,26,27,34,35,43) or counts.shape!=(anchors,) or ((counts<1)|(counts>10)).any() or
                        ((owners<0)|(owners>=anchors)).any() or not torch.equal(torch.bincount(owners,minlength=anchors),counts)):
                    raise RuntimeError("Variable Anchor members must have valid owners and capacity 1..10")
            elif width not in (11,26,27,34,35,43) or state["anchors.rows"].numel() != 10*anchors or not torch.equal(
                    state["anchors.rows"], torch.arange(anchors, device=state["anchors.rows"].device).repeat_interleave(10)):
                raise RuntimeError("Explicit Anchor checkpoint must contain ten Gaussian slots per anchor")
    else:
        model = SelectorNetwork().to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "selector checkpoint keys do not match the selector network: "
            "missing={}, unexpected={}".format(missing, unexpected)
        )
    model.eval()
    return model


def load_selector_metadata(path, device):
    """Load convergence state and saved per-row Bernoulli uniforms."""
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    comments = dict(line.split(" ", 1) for line in ply.comments if " " in line)
    if comments.get("protection_schema") != "continuous_cycle_geometric":
        raise RuntimeError("checkpoint requires continuous_cycle_geometric maturity metadata")
    metadata = {name: torch.as_tensor(
        np.asarray(vertex[name], dtype=np.float32).copy(), device=device)
        for name in ("maturity", "drift_ema", "cached_uniform")}
    for name, value in metadata.items():
        if not torch.isfinite(value).all():
            raise ValueError("non-finite maturity checkpoint field: "+name)
    if ((metadata["maturity"] < 0).any() or (metadata["maturity"] > 1).any()
            or (metadata["drift_ema"] < 0).any()
            or (metadata["cached_uniform"] < 0).any()
            or (metadata["cached_uniform"] >= 1).any()):
        raise ValueError("maturity checkpoint field outside its domain")
    metadata["mapping_iteration"] = int(comments["protection_mapping_iteration"])
    return metadata


def mask_inactive_slots(model, probability, opacity):
    """Same reversible eligibility as native; keep raw MLP/CD targets unchanged."""
    anchors = getattr(model, "anchors", None)
    if anchors is None or not hasattr(anchors, "slot_inactive"):
        return probability
    with torch.no_grad():
        low = opacity.detach().reshape_as(anchors.slot_inactive) < 1/255
        recovered = anchors.slot_inactive & ~low
        anchors.slot_recovered.logical_or_(recovered)
        anchors.slot_inactive.logical_and_(~recovered)
        eligible = ~(anchors.slot_inactive & low).flatten()
    return probability * eligible.to(probability.dtype)

@torch.no_grad()
def build_selector_mask(model, gaussians, metadata, ratio, config, temperature=1.0,
                        protection_enabled=True, forward_features=None):
    """All-row continuous handoff with the same saved uniforms as C++."""
    total = gaussians.get_xyz.shape[0]
    if metadata is None or any(metadata[k].numel() != total for k in
                               ("maturity", "drift_ema", "cached_uniform")):
        raise RuntimeError("continuous maturity metadata must match Gaussian count")
    c = classify_protection(metadata, config, protection_enabled)
    target_ratio = torch.full((total, 1), float(ratio),
                              dtype=torch.float32, device=gaussians.get_xyz.device)
    entry = config.get("entry_maturity", 0.0)
    def forward(rows):
        if forward_features is not None:
            if rows is not None:
                raise RuntimeError('Cached shared features require the all-row fixed-maturity path')
            return forward_features[0]
        if isinstance(model, AnchorSelectorNetwork):
            if model.gaussian_attributes:
                take = lambda value: value if rows is None else value[rows]
                return model(take(gaussians.get_xyz), None, take(gaussians.get_scaling), target_ratio[:1],
                             tau=temperature, source_rows=rows, sh_dc=take(gaussians.get_features[:, :1]),
                             opacity=take(gaussians.get_opacity))
            if rows is None:
                return model(gaussians.get_xyz, None, None, target_ratio[:1], tau=temperature)
            return model(gaussians.get_xyz[rows], None, None, target_ratio[:1], tau=temperature, source_rows=rows)
        if rows is None:
            return model(gaussians.get_xyz, gaussians.get_rotation, gaussians.get_scaling, target_ratio, tau=temperature)
        return model(gaussians.get_xyz[rows], gaussians.get_rotation[rows], gaussians.get_scaling[rows],
                     target_ratio[rows], tau=temperature)
    probability = elastic_selector_probability(metadata["maturity"], c["maturity"], ratio, entry, forward)
    probability = mask_inactive_slots(model, probability, gaussians.get_opacity)
    mask = metadata["cached_uniform"] < probability
    active, target = int(mask.sum().item()), ratio*total
    stats = protection_statistics(metadata, c)
    stats.update(target_active=target, expected_active=float(probability.sum().item()),
        actual_active=active, active_gaussians=active,
        actual_vs_target=active/max(1.0, target), ratio=float(ratio),
        protection_enabled=bool(protection_enabled),
        current_mapping_iteration=metadata["mapping_iteration"])
    eligible = int((metadata["maturity"] >= entry).sum().item())
    stats.update(elastic_entry_threshold=entry, elastic_eligible_count=eligible,
        elastic_eligible_fraction=eligible/max(1,total), selector_candidate_count=eligible,
        selector_skipped_low_maturity_count=total-eligible,
        maturity_below_0_3_fraction=float((metadata["maturity"] < .3).float().mean().item()),
        maturity_above_equal_0_3_fraction=float((metadata["maturity"] >= .3).float().mean().item()))
    if hasattr(getattr(model, "anchors", None), "slot_inactive"):
        inactive = int(model.anchors.slot_inactive.sum().item())
        stats.update(slot_deactivation=True, inactive_slots=inactive, slot_eligible_count=total-inactive,
                     budget_population=total, selector_candidate_count=total-inactive)
    return mask.float(), stats


@torch.no_grad()
def compensated_attributes(model, gaussians, selected_mask, ratio, shared=None):
    spatial=getattr(model,'spatial_compensation',None)
    if not hasattr(model,'compensation') and spatial is None:
        return None
    if spatial is not None and not bool(spatial.normalization_ready):
        return None
    from transform_field import _normalize_quaternion, _multiply_quaternion
    ids=torch.nonzero(selected_mask>0).flatten()
    if not len(ids):
        return None
    if spatial is not None:
        owners=model.anchors.rows[ids]
        residual=spatial(gaussians.get_xyz[ids],gaussians.get_scaling[ids],gaussians.get_rotation[ids],
            gaussians.get_features_dc[ids],gaussians._opacity[ids],model.anchors.centers[owners],
            model.anchors.offset_scales[owners],ratio)
    elif shared is None:
        _,shared=model.forward_selection_features(gaussians.get_xyz,gaussians.get_scaling,
            gaussians.get_features[:,:1],gaussians.get_opacity,
            torch.tensor([[ratio]],device=gaussians.get_xyz.device))
    if spatial is None:
        comp_input=shared[ids]
        if hasattr(model,'anchor_ratio_compensation_schema'):
            comp_input=torch.cat((comp_input,model.anchors.compensation_lookup(ids,ratio)),dim=1)
        residual=model.compensation(comp_input)
    if hasattr(model, 'gaussian_compensation'):
        residual = residual + model.gaussian_compensation(ids, ratio)
    xyz,scale,rotation=gaussians.get_xyz.clone(),gaussians.get_scaling.clone(),gaussians.get_rotation.clone()
    xyz[ids]+=residual[:,:3]
    scale[ids]*=residual[:,3:6].exp()
    identity=torch.zeros_like(residual[:,6:10]);identity[:,0]=1
    rotation[ids]=_normalize_quaternion(_multiply_quaternion(_normalize_quaternion(identity+residual[:,6:10]),rotation[ids]))
    result=dict(xyz=xyz,scaling=scale,rotation=rotation)
    if residual.shape[1]==14:
        logits,dc=gaussians._opacity.clone(),gaussians.get_features_dc.clone()
        logits[ids]+=residual[:,10:11]
        dc[ids]+=residual[:,11:14].unsqueeze(1)
        result.update(opacity_logits=logits,features_dc=dc)
    return result
