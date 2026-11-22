"""开发场景完成后锁定候选，不根据21组保留验证修改算法。"""
from common import *
path=HERE/'reports/FROZEN_CANDIDATE.json';assert not path.exists()
names=['source/cpu_small.cpp','source/dense_small.h','source/native_handoff.h','small_runtime.py','common.py','formal.py']
pool=lambda s:s[s.index('class Pool {'):s.index('// 权重以8个输出')]
assert pool((HERE/'source/cpu_small.cpp').read_text())==pool((BASE/'source/cpu_deep.cpp').read_text())
save(path,dict(frozen_at=time.time(),binary=extension().__file__,binary_sha256=sha(extension().__file__),
    sources={name:sha(HERE/name) for name in names},settings=dict(small_kernel=True,ordered_mode=3,threads=6),
    selection_cost_model='local: K*log2(max(2,K)) > M+4096选择顺序压缩，否则保留相同排序；global按阈值顺序压缩，保留ties',
    discarded=['缩短自旋','worker绑定','作用域主线程绑定','FP32两部分累加'],pool_identical_to_baseline=True,
    production_deployed=False,commit_push=False))
print('SMALL_FROZEN',sha(extension().__file__),flush=True)
