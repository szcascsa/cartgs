"""最终调优结束后冻结，后续不根据保留验证集改候选。"""
from common import *
path=HERE/'reports/FROZEN_CANDIDATE.json';assert not path.exists()
names=['source/cpu_deep.cpp','source/dense_fast.h','source/native_handoff.h','deep_runtime.py','common.py','formal.py','handoff_test.py']
save(path,dict(frozen_at=time.time(),binary=extension().__file__,binary_sha256=sha(extension().__file__),sources={name:sha(HERE/name) for name in names},
    final_variant='combined',kernel_mode=1,fused_schedule=True,native_wrapper=True,threads=6,
    rejected_candidate='8x8：没有稳定收益且小K退化，保留源码和tune02证据',production_deployed=False,commit_push=False))
print('DEEP_CANDIDATE_FROZEN',sha(extension().__file__),flush=True)
