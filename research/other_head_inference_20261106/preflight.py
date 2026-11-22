"""只读复用既有模型，独立复制其已验证Python源码。"""
import json
from pathlib import Path
import shutil
from common import ROOT,PREVIOUS,CHECKPOINT,sha,save

assert ROOT==Path('/home/prmi/zwj/experiments/other_head_inference_20261106')
assert not (ROOT/'reports/INPUTS_BEFORE.json').exists()
assert shutil.disk_usage(ROOT).free>5*1024**3
assert json.loads((PREVIOUS/'reports/COMPLETION.json').read_text())['integrity_passed']
manifest=json.loads((PREVIOUS/'reports/SOURCE_FINAL_HASHES.json').read_text())
source={}
for name,value in manifest.items():
    if name.startswith('python/') and name.endswith('.py'):
        p=PREVIOUS/'source'/name;assert sha(p)==value,name
        target=ROOT/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,target);assert sha(target)==value
        source[name]=value
save(ROOT/'reports/INFERENCE_SOURCE_HASHES.json',source)
inputs={str(p):sha(p) for p in CHECKPOINT.rglob('*') if p.is_file()}
for p in [PREVIOUS/'evaluation/reports/PER_VIEW.csv',PREVIOUS/'evaluation/reports/SUMMARY.json',
          PREVIOUS/'reports/SOURCE_FINAL_HASHES.json',
          Path('/home/prmi/zwj/experiments/fr1xyz_viewer_budget_comparison_20261105/build/fr1xyz_production_counts.so')]:
    inputs[str(p)]=sha(p)
for p in (PREVIOUS/'source/python').rglob('*.py'):inputs[str(p)]=sha(p)
save(ROOT/'reports/INPUTS_BEFORE.json',inputs)
names=json.loads((PREVIOUS/'reports/PRODUCTION_BEFORE.json').read_text())
save(ROOT/'reports/PRODUCTION_BEFORE.json',{n:sha(n) for n in names})
save(ROOT/'reports/PREFLIGHT.json',dict(passed=True,checkpoint=str(CHECKPOINT),source_files=len(source),
    protected_production_files=len(names),input_files=len(inputs),free_GiB=shutil.disk_usage(ROOT).free/1024**3))
print('独立源码、只读checkpoint及生产完整性基准已就绪',flush=True)
