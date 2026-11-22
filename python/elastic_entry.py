"""Elastic compute entry; raw maturity and global admission budget stay unchanged."""
import math
import torch
from elastic_budget import mix_selector_probability

def elastic_maturity(maturity, entry):
    if not math.isfinite(entry) or not 0 <= entry < 1:
        raise ValueError('Elastic.entry_maturity must be in [0,1)')
    if entry == 0:
        return maturity.detach()
    return ((maturity.detach()-entry)/(1-entry)).clamp(0,1)

def elastic_selector_probability(maturity, legacy_maturity, ratio, entry, forward):
    if entry == 0:
        return mix_selector_probability(forward(None),legacy_maturity,ratio)
    rows=torch.nonzero(maturity.detach() >= entry,as_tuple=False).flatten()
    probability=torch.full_like(maturity,ratio,requires_grad=False)
    if rows.numel():
        probability=probability.index_copy(0,rows,mix_selector_probability(
            forward(rows),elastic_maturity(maturity[rows],entry),ratio))
    return probability
