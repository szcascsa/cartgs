"""验证 C++/Python 可变成员 checkpoint、概率与 Cached Bernoulli 选择一致。"""
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(sys.argv[1])/'python'))
from selector import load_selector, mask_inactive_slots

device = sys.argv[2]
model = load_selector('variable_selector.pt', device)
fixture = torch.jit.load('variable_fixture.pt', map_location=device).state_dict()
owners = model.anchors.rows
counts = model.anchors.member_counts
assert torch.equal(torch.bincount(owners, minlength=len(counts)), counts)
results = []
for j, ratio in enumerate([.01, .05, .1, .15]):
    with torch.no_grad():
        p = model(fixture['xyz'], None, fixture['scale'], torch.tensor([[ratio]], device=device),
                  sh_dc=fixture['dc'], opacity=fixture['opacity'])
        p = mask_inactive_slots(model, p, fixture['opacity'])
    expected = fixture['p'+str(j)]
    error = float((p-expected).abs().max())
    assert torch.allclose(p, expected, atol=2e-6, rtol=2e-5), error
    assert torch.equal(p > fixture['uniform'], expected > fixture['uniform'])
    results.append(dict(ratio=ratio, max_probability_error=error, selected=int((p>fixture['uniform']).sum())))
Path('python_parity.json').write_text(json.dumps(dict(passed=True, results=results), indent=2))
print('PASS variable C++/Python checkpoint and Cached Bernoulli parity')
