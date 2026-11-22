"""隔离诊断：校准 sigmoid 的 Jacobian、gauge、centering 和局部平滑目标。

不读取地图/GT，不训练；结果仅证明反向传播性质，不代表 render 质量收益。
"""
import json
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)


def offset(score, k):
    with torch.no_grad():
        low, high = score.min().item() - 40, score.max().item() + 40
        for _ in range(80):
            mid = (low + high) / 2
            if (score - mid).sigmoid().sum().item() > k:
                low = mid
            else:
                high = mid
        return (low + high) / 2


def probability(score, k):
    return (score - offset(score, k)).sigmoid()


score = torch.tensor([-3., -1., -.2, .4, 1.3, 2.7], requires_grad=True)
k = 2
upstream = torch.tensor([0., -1.4, 0., .2, -.7, 0.])
p = probability(score, k)
d = p.detach() * (1 - p.detach())
g = torch.autograd.grad((p * upstream).sum(), score)[0]
g_center = g - g.mean()
g_calibrated = g - d * g.sum() / d.sum()
jacobian = torch.diag(d) - torch.outer(d, d) / d.sum()
eps = 1e-5
fd = []
for i in range(len(score)):
    delta = torch.zeros_like(score)
    delta[i] = eps
    fd.append(((probability(score.detach() + delta, k) * upstream).sum()
               - (probability(score.detach() - delta, k) * upstream).sum()) / (2 * eps))
fd = torch.stack(fd)
assert torch.max(torch.abs(g_calibrated - fd)).item() < 1e-8
assert torch.allclose(g_calibrated, jacobian @ upstream)
assert abs(g.sum().item()) > 1e-3
assert abs(g_center.sum().item()) < 1e-14
assert torch.max(torch.abs(g_center - g_calibrated)).item() > 1e-3

c = torch.tensor(0., requires_grad=True)
bad_shift_grad = torch.autograd.grad((probability(score.detach() + c, k) * upstream).sum(), c)[0]
r = score.detach().clone().requires_grad_()
centered = r - r.mean()
center_grad = torch.autograd.grad((probability(centered, k) * upstream).sum(), r)[0]
torch.testing.assert_close(center_grad, g_center)
shift_forward_error = (probability(score.detach() + 100, k) - p.detach()).abs().max().item()

# 固定本地 opacity upstream；比较“梯度下降”对真正重新校准的连续局部目标的方向。
# 这是 smooth-proxy 有限差分，不把离散 Top-K 的有限差分当作 ST 正确性标准。
def local_objective(s):
    return float((probability(s, k) * upstream).sum())

directional_changes = {}
for name, grad in [('detached', g), ('centered', g_center), ('calibrated', g_calibrated)]:
    directional_changes[name] = (local_objective(score.detach() - eps * grad)
                                 - local_objective(score.detach())) / eps

# 存在某个合法 upstream，使 detached/centered surrogate 与校准目标的梯度方向相反。
# 确定性查找只用于构造反例，不用于选择任何模型超参数。
generator = torch.Generator().manual_seed(20261020)
counterexample = None
for _ in range(10000):
    s = torch.randn(6, generator=generator) * 3
    h = torch.randn(6, generator=generator)
    q = probability(s, k)
    dd = q * (1 - q)
    gg = h * dd
    truth = gg - dd * gg.sum() / dd.sum()
    cent = gg - gg.mean()
    if torch.dot(gg, truth) < -1e-5:
        actual_fd = (float((probability(s - eps * gg, k) * h).sum())
                     - float((probability(s, k) * h).sum())) / eps
        counterexample = dict(score=s.tolist(), upstream=h.tolist(),
                              detached_dot_true=float(torch.dot(gg, truth)),
                              centered_dot_true=float(torch.dot(cent, truth)),
                              actual_directional_FD=actual_fd)
        assert actual_fd > 0
        break
assert counterexample is not None

result = dict(all_passed=True, scope='FP64 CPU toy; no renderer, no quality claim',
              score=score.detach().tolist(), upstream=upstream.tolist(), probability=p.detach().tolist(),
              detached_gradient=g.tolist(), centered_gradient=g_center.tolist(),
              calibrated_gradient=g_calibrated.tolist(), finite_difference=fd.tolist(),
              calibrated_FD_max_error=float((g_calibrated - fd).abs().max()),
              centered_calibrated_L2_gap=float((g_center - g_calibrated).norm()),
              common_shift_forward_max_error=shift_forward_error,
              detached_common_shift_gradient=float(bad_shift_grad),
              centered_common_shift_gradient=float(center_grad.sum()),
              descent_directional_FD=directional_changes, counterexample=counterexample)
Path(__file__).with_name('toy_gradients.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
