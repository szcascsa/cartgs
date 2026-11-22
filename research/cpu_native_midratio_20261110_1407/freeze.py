"""三个开发场景确定后冻结，不依保留验证场景调参。"""
from common import *
path=HERE/'reports/FROZEN_CANDIDATE.json';assert not path.exists()
names=['source/cpu_mid.cpp','source/native_handoff.h','mid_runtime.py','common.py','formal.py']
pool=lambda s:s[s.index('class Pool {'):s.index('// 权重以8个输出')]
assert pool((HERE/'source/cpu_mid.cpp').read_text())==pool((BASE/'source/cpu_small.cpp').read_text())
save(path,dict(frozen_at=time.time(),binary=extension().__file__,binary_sha256=sha(extension().__file__),sources={n:sha(HERE/n) for n in names},
    primary_ratios=[.1,.2,.3,.4,.5],settings=dict(selection_mode=1,comp_batch=6,threads=6),
    selection=dict(bins=256,minimum_M=2048,max_boundary_fraction=.25,fallback='原nth_element与ordered_mode=3'),
    numerical_contract='Score、UID、非Rotation属性逐位一致；Rotation绝对误差<=1e-6并独立渲染质量筛查',
    baseline=str(BASE),pool_unchanged=True,production_deployed=False,commit_push=False))
protocol=json.loads((HERE/'reports/PROTOCOL.json').read_text());protocol['numerical_contract']='Score/UID/非Rotation逐位一致，Rotation<=1e-6；最终GT PSNR平均下降<=.05dB、各条件平均<=.2dB，报告最差帧'
save(HERE/'reports/PROTOCOL.json',protocol)
print('MID_FROZEN',sha(extension().__file__),flush=True)
