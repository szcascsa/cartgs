"""全部质量输出完成后的研究判读；不修改训练、LR、checkpoint或selected sets。"""
from pathlib import Path
import csv, json

ROOT = Path(__file__).resolve().parent
path = ROOT / 'audit_summary.json'
s = json.loads(path.read_text())
assert s['correctness']['all_passed']
assert s['correctness']['formal_updates'] == 96000
assert s['counts']['Grid-MLP']['Raw']['eval']['psnr']['improved'] == 5
assert s['counts']['Independent-Cell']['Raw']['eval']['psnr']['improved'] == 9
gate_final = next(z for z in csv.DictReader(open(ROOT/'quality_deltas.csv')) if z['arm']=='Grid-MLP' and z['reference']=='Raw' and z['split']=='train' and z['scene']=='fr1_xyz' and float(z['ratio'])==.05 and int(z['seed'])==2026)
assert float(gate_final['photo']) > 0
s['decision'] = 'D. REGIONAL CORRECTION NOT SUPPORTED'
s['interpretation'] = (
    '在本轮固定grid、calibrated ST、统一LR与2000步协议下，Grid-MLP未获得跨scene/ratio/seed稳定收益。'
    'non-Bank相对Raw仅5/24个case的PSNR改善，平均ΔPSNR=-0.441525 dB，photo仅6/24改善；'
    'Independent-Cell为9/24个PSNR改善、平均ΔPSNR=-0.001689 dB，photo同样仅6/24改善。'
    '两组均24/24发生真实cell间预算转移，因此不能用coefficient非零或budget movement判成功。'
    '当前不建议进入gradient-threshold split。')
s['answers'] = dict(Q1='YES', Q2='YES', Q3='YES', Q4='NO', Q5='NO',
                    Q6='NO', Q7='NO', Q8='NO', Q9='INCONCLUSIVE', Q10='NO-GO')
s['mechanism_interpretation'] = [
    'Grid-MLP的负结果主要集中在fr1_xyz与fr2_xyz：两scene各0/8个non-Bank PSNR case优于Raw；fr1_room为5/8。不能由room的少量改善外推跨scene有效。',
    'Independent-Cell整体明显减轻Grid-MLP的损失，但未稳定优于Raw；因此不满足“Independent-Cell明显成功、MLP失败”的C类前提。fr1_xyz的局部正向信号仍保留在完整表中，没有被全局平均掩盖。',
    'Q8的NO表示本轮未证明grid-level parameterization能稳定提高quality，不表示数学上不存在更好的cell coefficients。两种参数化共享当前ST目标；Independent-Cell也复用Grid-MLP选出的LR。当前实验不能隔离regional representation、MLP sharing和optimization/objective各自的因果份额，所以Q9为INCONCLUSIVE。',
    'LR sanity的3e-4在step250改善Bank photo，但该模型训练到step2000后变差；短期gate通过不能替代固定预算的最终hard quality。本轮保持预注册的2000步，没有事后选择step250或扩大LR搜索。',
    '加权gauge与calibrated VJP正确并不保证ST training objective会提高最终deterministic hard Top-K。当前结果是对这一固定训练协议的负证据，不能归因于common-score-shift错误仍存在，也不能单独证明grid表达能力不足。',
    '下一步决策为NO-GO：本轮不进入adaptive grid/split，也不据gradient统计调整threshold。若继续研究，应先解释固定regional参数化下hard质量为何无法稳定改善；本轮不自动执行后续实验。'
]
path.write_text(json.dumps(s, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
print(s['decision'])
