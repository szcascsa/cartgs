"""保留整程序racecheck失败，并用原版对照与新增Kernel过滤检查定位来源。"""
from pathlib import Path
import json
import re

HERE = Path(__file__).resolve().parent
R = HERE / 'reports'


def job(name):
    return json.loads((R / 'jobs' / (name + '.json')).read_text())


def races(name):
    log = (HERE / 'logs' / (name + '.log')).read_text()
    return re.findall(r'========= Error: Race reported.*?\[6144 hazards\]', log, re.S)


assert job('memcheck01')['returncode'] == 0
assert job('synccheck01')['returncode'] == 0
assert job('racecheck01')['returncode'] == 97
assert job('race_control01')['returncode'] == 97
assert job('racecheck_tensor02')['returncode'] == 0
current, baseline = races('racecheck01'), races('race_control01')
assert len(current) == 2 and current == baseline
assert all('at::native::reduce_kernel' in item and 'ReduceOp<long' in item
           for item in current)
control = json.loads((R / 'race_control01/SUMMARY.json').read_text())
assert control['passed'] and control['baseline_only']
result = dict(
    attribution_passed=True, whole_program_racecheck_passed=False,
    whole_program_race_errors=2, unchanged_baseline_race_errors=2,
    reported_races_identical=True,
    reported_kernel='at::native::reduce_kernel<long sum>',
    read_pc='0x9280', write_pc='0x9370', hazards_per_report=6144,
    baseline_control=control,
    tensor_kernel_racecheck_passed=True,
    tensor_filters=['kns=selectorTensor', 'kns=compensationTensor', 'kns=packTensorWeights'],
    filter_scope=['packTensorWeights', 'selectorTensor', 'compensationTensor'],
    limitation='整程序未零告警；相同ATen告警已在未加载Tensor扩展的原v31复现。'
               '未判定其为工具误报，也未修改共享PyTorch。新增Tensor Kernel过滤检查零告警。',
)
(R / 'SANITIZER_ATTRIBUTION.json').write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
