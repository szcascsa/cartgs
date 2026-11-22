"""精确后端的缓存边界、异常检查及最终源码/服务器完整性核验。"""
from pathlib import Path
import hashlib
import json
import os
import sys
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source/python'))
from budget_viewer import BudgetScene, read_config
from anchor_budget import FixedBudgetRenderer

torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32 = False
scene = BudgetScene('/home/prmi/zwj/realLOD/results/fr1_xyz_frozen_master_20261103_01/r20_40_60_80/training/13052_shutdown')
model, pc = scene.model, scene.gaussians
renderer = FixedBudgetRenderer(model, pc, read_config(scene.checkpoint/'selector_config.yaml'), backend='cuda_graph')
checks = []


@torch.no_grad()
def run():
    engine = renderer.inference
    initial = torch.cuda.memory_allocated()
    for k in (1, 17, 200, 700, 1700, 2560, 4000, 5000, 6000, 8000, 10000, 12000, 17):
        rows = torch.arange(k, device='cuda')
        for ratio in (.2, .3, .4):engine.render_data(rows, ratio)
        assert len(engine.graphs) <= 6 and engine.graph_bytes <= 512*1024**2
    checks.append('13种实际K的LRU淘汰及显存边界通过')
    controls = engine.controls(.4)
    rows = torch.arange(100, device='cuda')
    for value in (float('nan'), float('inf'), -float('inf')):
        saved = model.anchors.compensation_features_0.clone()
        model.anchors.compensation_features_0.fill_(value)
        renderer.rebuild();engine = renderer.inference
        try:engine.render_data(rows, .4)
        except RuntimeError as error:assert 'Nonfinite' in str(error)
        else:raise AssertionError('Nonfinite Feature was accepted')
        model.anchors.compensation_features_0.copy_(saved);renderer.rebuild()
    checks.append('非有限Feature保持报错')
    try:renderer.inference.ext.selector_logits(pc.get_xyz.new_zeros((1, 11)), renderer.inference.selector_parameters)
    except RuntimeError:raise AssertionError('no_grad path failed')
    with torch.enable_grad():
        try:renderer.inference.ext.selector_logits(pc.get_xyz.new_zeros((1, 11)), renderer.inference.selector_parameters)
        except RuntimeError as error:assert 'Frozen' in str(error)
        else:raise AssertionError('Training silently entered frozen kernel')
    checks.append('训练GradMode不能误入冻结后端')
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    integrity = {}
    for prefix, base in (('production', Path('/home/prmi/zwj/mycartgs')), ('realLOD', Path('/home/prmi/zwj/realLOD'))):
        before = json.loads((ROOT/(prefix+'_before.json')).read_text())
        after = {name: sha(base/name) if (base/name).is_file() else None for name in before}
        changed = [name for name in before if before[name] != after[name]]
        integrity[prefix] = dict(files=len(before), unchanged=not changed, changed=changed)
        (ROOT/(prefix+'_after.json')).write_text(json.dumps(after, indent=2)+'\n')
    files = ['python/frozen_inference_cuda.py', 'python/frozen_inference_bindings.cpp',
             'python/frozen_inference_kernels.cu', 'python/anchor_budget.py',
             'python/budget_viewer.py', 'scripts/view_budget.py']
    manifest = {name: sha(ROOT/'source'/name) for name in files}
    (ROOT/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
    result = dict(passed=all(value['unchanged'] for value in integrity.values()), checks=checks,
        graph_allocated_delta=torch.cuda.memory_allocated()-initial, integrity=integrity,
        torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
        tf32=torch.backends.cuda.matmul.allow_tf32)
    (ROOT/'FINAL_CHECKS.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


run()
