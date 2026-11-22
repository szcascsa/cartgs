"""普通背景优化验证通过后，独立冻结并明确开发集变更。"""
from render_study import *
import formal
from background_runtime import extension

formal.frozen()
formal.identities()
stage=HERE/'background_stage'
out=stage/'reports/FROZEN.json'
assert not out.exists()
assert json.loads((stage/'reports/validation01/BACKGROUND_PASS.json').read_text())['passed']
sources={str(p.relative_to(stage)):sha(p) for p in (stage/'source').rglob('*') if p.is_file()}
for n in ('formal.py',):
    sources[n]=sha(stage/n)
for n in ('background_runtime.py','render_study.py','lifecycle.py','background_validation.py'):
    sources['../'+n]=sha(HERE/n)
tune=TUNE+['VECtor__corridors-dolly__stereo']
binary=Path(extension().__file__)
save(out,dict(frozen=True,mode=18,binary=str(binary),binary_sha256=sha(binary),sources=sources,
              tune=tune,heldout=[s['id'] for s in SCENES if s['slug'] not in tune],frozen_at=time.time(),
              change='仅跳过普通有限normal/+0背景的冗余空成员全图扫描；敏感背景保留原证明路径',
              heldout_revision='corridors-dolly双目因第一阶段发现反例转为开发场景，其余27组不调参'))
print('BACKGROUND_FROZEN',binary,flush=True)
