"""仅开发集决定候选；冻结之后才运行保留场景。"""
from render_study import *

out=HERE/'reports/FROZEN.json'
assert not out.exists()
assert json.loads((HERE/'reports/lifecycle03/SUMMARY.json').read_text())['passed']
sources={str(p.relative_to(HERE)):sha(p) for p in (HERE/'source').rglob('*') if p.is_file()}
for n in ('scaling_runtime.py','render_study.py','formal.py','lifecycle.py'):
    sources[n]=sha(HERE/n)
binary=Path(extension().__file__)
save(out,dict(frozen=True,mode=18,name=MODES[18],binary=str(binary),binary_sha256=sha(binary),sources=sources,
              tune=TUNE,heldout=[s['id'] for s in SCENES if s['slug'] not in TUNE],
              rule='同一顺序FP32 Alpha，无Ratio分流或人为延时；小容量稳定整数排序和32次原顺序循环展开',
              selected_from=['screen03','screen04','screen05'],frozen_at=time.time()))
print('CANDIDATE_FROZEN',MODES[18],binary,flush=True)
