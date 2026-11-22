"""同一GT跨22次渲染复用LPIPS目标特征；沿用原TorchMetrics全部公式。"""
from collections import OrderedDict
import torch
from torchmetrics.functional.image.lpips import _normalize_tensor, _spatial_average


class CachedTargetLPIPS:
    def __init__(self, metric):
        self.net = metric.net
        assert not self.net.training and self.net.resize is None and not self.net.spatial
        self.cache = OrderedDict()

    @torch.no_grad()
    def __call__(self, prediction, truths, frames):
        target_features = []
        for truth, frame in zip(truths, frames):
            if frame not in self.cache:
                output = self.net.net.forward(self.net.scaling_layer(2*truth[None]-1))
                self.cache[frame] = tuple(_normalize_tensor(output[layer]) for layer in range(self.net.L))
                if len(self.cache) > 4:
                    self.cache.popitem(last=False)
            target_features.append(self.cache[frame])
        output = self.net.net.forward(self.net.scaling_layer(2*prediction-1))
        residuals = []
        for layer in range(self.net.L):
            target = torch.cat([features[layer] for features in target_features])
            difference = (_normalize_tensor(output[layer])-target)**2
            residuals.append(_spatial_average(self.net.lins[layer](difference), keep_dim=True))
        return sum(residuals).reshape(-1)
