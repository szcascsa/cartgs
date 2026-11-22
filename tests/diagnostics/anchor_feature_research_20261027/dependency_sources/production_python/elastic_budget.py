"""Continuous geometric maturity handoff; no Protection admission budgets."""
import math
import torch

DEFAULTS = dict(drift_temperature=0.10, beta_fast=0.60, beta_slow=0.15,
                compatibility_mature_threshold=0.80)

def validate_config(config):
    t, fast, slow, threshold = (config[k] for k in DEFAULTS)
    if (not all(math.isfinite(v) for v in (t, fast, slow, threshold)) or
            t <= 0 or not 0 < slow < fast <= 1 or not 0 <= threshold <= 1):
        raise ValueError("invalid continuous geometric maturity configuration")

@torch.no_grad()
def classify_protection(metadata, config, enabled=True):
    validate_config(config)
    maturity = metadata["maturity"].detach()
    if not enabled:
        maturity = torch.ones_like(maturity)
    return dict(maturity=maturity,
                mature=maturity >= config["compatibility_mature_threshold"])

def mix_selector_probability(probability, maturity, ratio):
    if not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("invalid global ratio")
    if probability.shape != maturity.shape:
        raise ValueError("maturity is not row-aligned")
    m = maturity.detach()
    return (1 - m) * ratio + m * probability

def sample_selector_st(probability, uniform):
    if probability.shape != uniform.shape:
        raise ValueError("uniform is not row-aligned")
    hard = (uniform < probability).to(probability.dtype)
    return hard + (probability - probability.detach())

def ratio_loss(probability, ratio):
    target = ratio * probability.numel()
    return (probability.sum() - target).abs() / max(1.0, target)

@torch.no_grad()
def protection_statistics(metadata, classification):
    m = classification["maturity"]
    stats = {"total_gaussians": m.numel(),
             "compatibility_mature_count": int(classification["mature"].sum().item())}
    for name, value, quantiles in (
            ("maturity", m, (0.1, 0.5, 0.9)),
            ("drift", metadata["drift_ema"], (0.5, 0.9, 0.95))):
        stats[name+"_mean"] = float(value.mean().item()) if value.numel() else 0.0
        for q in quantiles:
            stats[name+"_p"+str(round(q*100))] = float(torch.quantile(value, q).item()) if value.numel() else 0.0
        stats[name+"_median"] = stats[name+"_p50"]
    for i in range(5):
        bucket = (m >= i*0.2) & ((m <= 1) if i == 4 else (m < (i+1)*0.2))
        stats["maturity_bucket_"+str(i)] = int(bucket.sum().item())
    return stats
