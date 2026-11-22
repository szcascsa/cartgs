"""Evidence-aware classification and demand-driven Protection admission."""
import math
import torch

DEFAULTS = dict(age_ready=100, optimization_ready=80, unique_keyframe_ready=4,
                stale_age=1000, max_budget_fraction=0.35)

def validate_config(config):
    if any(config[k] <= 0 for k in ("age_ready", "optimization_ready", "unique_keyframe_ready")):
        raise ValueError("Protection evidence thresholds must be positive")
    if config["stale_age"] < config["age_ready"]:
        raise ValueError("Protection.stale_age must be >= age_ready")
    fraction = config["max_budget_fraction"]
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Protection.max_budget_fraction must be in [0,1]")

@torch.no_grad()
def classify_protection(metadata, config, enabled=True):
    validate_config(config)
    age = (metadata["mapping_iteration"] - metadata["birth_iter"]).clamp_min(0)
    a = (age.float() / config["age_ready"]).clamp(0, 1)
    u = (metadata["optimized_count"].float() / config["optimization_ready"]).clamp(0, 1)
    k = (metadata["unique_keyframe_count"].float() / config["unique_keyframe_ready"]).clamp(0, 1)
    maturity = torch.minimum(torch.minimum(a, u), k)
    mature = ((age >= config["age_ready"]) &
              (metadata["optimized_count"] >= config["optimization_ready"]) &
              (metadata["unique_keyframe_count"] >= config["unique_keyframe_ready"]))
    state = torch.ones_like(metadata["optimized_count"])
    state[age >= config["stale_age"]] = 2
    state[age < config["age_ready"]] = 0
    state[mature] = 3
    if not enabled:
        mature = torch.ones_like(mature)
        state.fill_(3)
        maturity = torch.ones_like(maturity)
    return dict(age=age, age_confidence=a, optimization_confidence=u,
                keyframe_confidence=k, maturity=maturity, need=1-maturity,
                state=state, mature=mature)

@torch.no_grad()
def protection_budget(mature, need, uniform, ratio, max_budget_fraction=0.35):
    if not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("invalid global ratio")
    if not math.isfinite(max_budget_fraction) or not 0 <= max_budget_fraction <= 1:
        raise ValueError("invalid Protection budget fraction")
    if mature.shape != need.shape or mature.shape != uniform.shape:
        raise ValueError("Protection population is not aligned")
    total = ratio * mature.numel()
    need = need * (~mature).to(need.dtype)
    demand = need.sum()
    target = demand.clamp_max(max_budget_fraction * total)
    # The requested budget always satisfies Bp <= D, so lambda <= 1.
    probability = need * target / demand.clamp_min(1e-12)
    admitted = (~mature) & (uniform < probability)
    actual = int(admitted.sum().item())
    remaining = max(0.0, min(float(mature.sum().item()), total-actual))
    return admitted, probability, dict(protection_demand=float(demand.item()),
        protection_budget_target=float(target.item()), protection_actual=actual,
        protected_actual=actual, mature_budget_target=remaining, target_total_gaussians=total)

@torch.no_grad()
def protection_statistics(metadata, classification):
    c = classification
    states = c["state"]
    mean = lambda x: float(x.float().mean().item()) if x.numel() else 0.0
    stale = states == 2
    stats = {name: int((states == index).sum().item()) for index, name in enumerate(
        ("warmup_gaussians", "under_observed_gaussians", "stale_gaussians", "mature_gaussians"))}
    stats.update(protected_gaussians=int((~c["mature"]).sum().item()),
        mean_maturity_confidence=mean(c["maturity"]), mean_protection_need=mean(c["need"]),
        mean_age_confidence=mean(c["age_confidence"]),
        mean_optimization_confidence=mean(c["optimization_confidence"]),
        mean_keyframe_confidence=mean(c["keyframe_confidence"]),
        mean_optimized_count=mean(metadata["optimized_count"]),
        mean_unique_keyframe_support=mean(metadata["unique_keyframe_count"]),
        stale_count=int(stale.sum().item()), stale_mean_age=mean(c["age"][stale]),
        stale_mean_optimized_count=mean(metadata["optimized_count"][stale]),
        stale_mean_unique_keyframes=mean(metadata["unique_keyframe_count"][stale]))
    for name in ("warmup_to_under_observed", "warmup_to_mature", "under_observed_to_mature",
                 "under_observed_to_stale", "stale_to_mature"):
        if name in metadata: stats[name] = metadata[name]
    return stats
