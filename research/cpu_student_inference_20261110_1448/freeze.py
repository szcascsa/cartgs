"""新功能及原后端身份固定；不改变已有编译产物。"""
from common import *
path=HERE/'reports/FROZEN_CANDIDATE.json';assert not path.exists()
assert json.loads((HERE/'reports/PREFLIGHT.json').read_text())['passed']
save(path,dict(frozen_at=time.time(),binary=extension().__file__,binary_sha256=sha(extension().__file__),
    sources={name:sha(HERE/name) for name in ['student_runtime.py','common.py','formal.py','quality.py']},
    backend_sources={str(p):sha(p) for p in [BASE/'source/cpu_mid.cpp',BASE/'source/native_handoff.h',BASE/'mid_runtime.py']},
    student_models={str(FROZEN/'models'/s['slug']/'continuous_calibration/best.pt'):sha(FROZEN/'models'/s['slug']/'continuous_calibration/best.pt') for s in SCENES},
    parameters='Teacher64/64两层Heads；Student32/32直接Heads，同一Selector/Feature/Master',
    default='Teacher；只有显式student_checkpoint或绑定Student实例才启用',
    newly_compiled=False,backend_reused=True,training_modified=False,production_deployed=False,commit_push=False))
print('STUDENT_FROZEN',sha(extension().__file__),flush=True)
