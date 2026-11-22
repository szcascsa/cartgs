"""在 Python Torch 自己的 ABI 下验证 inference 和完整一阶 autograd。"""
import json
from pathlib import Path
import sys

import torch

dest = Path(__file__).resolve().parent
root = dest.parents[2]
sys.path.insert(0, str(root/'python'))
from selector import load_selector, SelectorNetwork

checkpoint = '/home/prmi/zwj/mycartgs/results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply/selector.pt'
torch.manual_seed(123)
a = load_selector(checkpoint, 'cuda')
b = load_selector(checkpoint, 'cuda')
b.set_backend('cuda')
checks = []


def check(name, x, y):
    passed = torch.equal(x, y)
    checks.append(dict(name=name, exact=passed, max_abs=float((x-y).abs().max()) if x.numel() else 0))
    assert passed, checks[-1]


for n in (0, 1, 257, 219969):
    x = torch.randn(n, 3, device='cuda', requires_grad=True)
    q = torch.randn(n, 4, device='cuda', requires_grad=True)
    s = torch.randn(n, 3, device='cuda').exp().detach().requires_grad_(True)
    r = torch.full((n, 1), .05, device='cuda')
    pa, pb = a(x, q, s, r), b(x, q, s, r)
    check(f'n{n}/probability', pb, pa)
    if n:
        u = torch.rand(n, device='cuda')
        check(f'n{n}/hard', (u<pb).float(), (u<pa).float())
    dy = torch.randn_like(pa)
    ga = torch.autograd.grad(pa, [x, q, s, *a.parameters()], dy)
    gb = torch.autograd.grad(pb, [x, q, s, *b.parameters()], dy)
    for i, (xx, yy) in enumerate(zip(ga, gb)):
        check(f'n{n}/grad{i}', yy, xx)
teacher = SelectorNetwork().cuda()
teacher.load_state_dict(b.state_dict())
assert list(a.state_dict()) == list(b.state_dict()) and len(b.state_dict()) == 18
torch.save(b.state_dict(), dest/'python_selector_roundtrip.pt')
teacher.load_state_dict(torch.load(dest/'python_selector_roundtrip.pt'))
(dest/'python_parity.json').write_text(json.dumps(dict(torch=torch.__version__, abi=torch._C._GLIBCXX_USE_CXX11_ABI,
    checks=checks, all_exact=True, checkpoint_keys_unchanged=True), indent=2)+'\n')
print('Python CUDA Selector parity exact; checkpoint keys unchanged')
