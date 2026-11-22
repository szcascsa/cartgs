"""FlexGS采样、hard-mask预算及分位数推理；不改变Selector网络。"""
import math
import torch


def gumbel_selection(logits, temperature=1.0, hard=True, noise=None):
    if logits.ndim != 2 or logits.shape[1] != 2 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Gumbel selection requires Nx2 logits and a positive finite temperature')
    if noise is None:
        noise = -torch.empty_like(logits).exponential_().log()
    if noise.shape != logits.shape or noise.device != logits.device:
        raise ValueError('Gumbel noise must match Selector logits')
    soft = ((logits + noise) / temperature).softmax(-1)
    if not hard:
        return soft[:, 1]
    indices = soft.max(-1, keepdim=True).indices
    discrete = torch.zeros_like(logits).scatter_(-1, indices, 1.0)
    return (discrete - soft.detach() + soft)[:, 1]


def gumbel_ratio_loss(mask, ratio, normalize_target=False):
    if normalize_target:
        target = ratio * mask.numel()
        return (mask.sum() - target).abs() / max(1.0, target)
    return (mask.mean() - ratio).abs() if mask.numel() else mask.sum()


@torch.no_grad()
def flexgs_inference_mask(logits, ratio, temperature=1.0, eligible=None, noise=None):
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError('FlexGS inference ratio must be in (0,1]')
    scores = gumbel_selection(logits, temperature, hard=False, noise=noise)
    eligible = torch.ones_like(scores, dtype=torch.bool) if eligible is None else eligible.bool()
    if eligible.shape != scores.shape:
        raise ValueError('Eligibility must match Gaussian rows')
    scores = scores.masked_fill(~eligible, -float('inf'))
    if not scores.numel():
        return eligible, scores, None
    index = int((1 - ratio) * (scores.numel() - 1))
    threshold = scores.sort().values[index]
    return eligible & (scores >= threshold), scores, threshold
