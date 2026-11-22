"""冻结性能候选，之后仅验证保留场景。"""
from cpu_study import *
path=HERE/'reports/FROZEN_CANDIDATE.json';assert not path.exists()
required=['source/cpu_native.cpp','cpu_runtime.py','cpu_gpu_handoff.py','cpu_study.py','paired.py','quality.py','boundaries.py']
save(path,dict(frozen=True,time=time.time(),binary=extension().__file__,binary_sha256=sha(extension().__file__),
    sources={name:sha(HERE/name) for name in required},threads=6,selector_prefix=True,factored=True,
    precision='FP32 AVX2/FMA；首层分解改变浮点分组，允许经验证的微差',
    production=False,commit_push=False))
print('CPU_CANDIDATE_FROZEN',sha(extension().__file__))
