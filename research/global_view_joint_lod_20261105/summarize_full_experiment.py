"""根据配对逐帧结果生成中文结论；严格沿用训练前固定的质量门槛。"""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT/'evaluation/reports'
TRAIN = [.01, .05, .1, .3, .5, .7]
ALL = [.01, .03, .05, .075, .1, .2, .3, .4, .5, .6, .7]
labels = {'baseline': '原基线', 'joint': '全局＋局部联合', 'global_control': '额外全局对照'}
assert json.loads((ROOT/'evaluation/STATUS.json').read_text())['state'] == 'complete'
summary = json.loads((REPORTS/'SUMMARY.json').read_text())
index = {(r['group'], r['scope'], r['ratio'], r['partition']): r for r in summary}
raw = list(csv.DictReader((REPORTS/'PER_VIEW.csv').open()))
lookup = {(r['group'], r['case'], int(r['frame'])): r for r in raw}
deltas = []
for group in ['joint', 'global_control']:
    for row in raw:
        if row['group'] != group:
            continue
        baseline = lookup[('baseline', row['case'], int(row['frame']))]
        assert row['image'] == baseline['image'] and row['timestamp'] == baseline['timestamp']
        delta = {key: row[key] for key in ['case', 'scope', 'ratio', 'ratio_split', 'frame', 'bank']}
        delta['group'] = group
        for key in ['psnr', 'ssim', 'lpips', 'photo_l1', 'K', 'candidate_N', 'contributing_K']:
            delta['delta_'+key] = float(row[key])-float(baseline[key])
        delta['same_selection'] = row['selection_sha256'] == baseline['selection_sha256']
        deltas.append(delta)
with (REPORTS/'PAIRED_DELTAS.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(deltas[0])); writer.writeheader(); writer.writerows(deltas)
assert all(row['delta_K'] == 0 for row in deltas if row['scope'] == 'view'), '核对局部自然预算逐帧点数'
paired = []
for group in ['joint', 'global_control']:
    for case in sorted({r['case'] for r in deltas}):
        for partition in ['all', 'bank', 'nonbank']:
            rows = [r for r in deltas if r['group'] == group and r['case'] == case and
                    (partition == 'all' or r['bank'] == str(partition == 'bank'))]
            item = {key: rows[0][key] for key in ['group', 'case', 'scope', 'ratio', 'ratio_split']}
            item.update(partition=partition, frames=len(rows), psnr_improved_fraction=float(np.mean([r['delta_psnr'] > 0 for r in rows])),
                        same_K_fraction=float(np.mean([r['delta_K'] == 0 for r in rows])),
                        same_selection_fraction=float(np.mean([r['same_selection'] for r in rows])))
            for key in ['psnr', 'ssim', 'lpips', 'K', 'contributing_K']:
                values = np.asarray([r['delta_'+key] for r in rows])
                item.update({f'delta_{key}_{label}': float(value) for label, value in [
                    ('mean', values.mean()), ('median', np.median(values)), ('p10', np.percentile(values, 10)), ('p90', np.percentile(values, 90))]})
            paired.append(item)
(REPORTS/'PAIRED_SUMMARY.json').write_text(json.dumps(paired, indent=2))
with (REPORTS/'PAIRED_SUMMARY.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(paired[0])); writer.writeheader(); writer.writerows(paired)


def value(group, scope, ratio, metric='psnr', partition='all'):
    return index[(group, scope, ratio, partition)][metric+'_mean']


gates = {}
for group in ['joint', 'global_control']:
    global_delta = [value(group, 'global', ratio)-value('baseline', 'global', ratio) for ratio in TRAIN]
    mean_delta = float(np.mean(global_delta))
    gates[group] = {'global_train_mean_delta_psnr': mean_delta, 'global_train_min_delta_psnr': min(global_delta),
                    'mean_threshold': -.1, 'per_ratio_threshold': -.2,
                    'passed': mean_delta >= -.1 and min(global_delta) >= -.2,
                    'per_ratio_delta': dict(zip(map(str, TRAIN), global_delta)),
                    'view_train_mean_delta_psnr': float(np.mean([value(group, 'view', ratio)-value('baseline', 'view', ratio) for ratio in TRAIN])),
                    'view_unseen_mean_delta_psnr': float(np.mean([value(group, 'view', ratio)-value('baseline', 'view', ratio) for ratio in ALL if ratio not in TRAIN]))}
(REPORTS/'PREDECLARED_QUALITY_GATES.json').write_text(json.dumps(gates, indent=2))
audit = json.loads((ROOT/'reports/FULL_TRAINING_AUDIT.json').read_text())
status = json.loads((ROOT/'reports/FULL_EXPERIMENT_STATUS.json').read_text())
lines = ['# fr1_xyz Full checkpoint：全局＋局部联合 Fine 结果', '',
    '本轮为 RGB-D、seed0 的固定拓扑 Full checkpoint 继续训练。三组完整 Fine 及792共同RGB帧的全轨迹评价均完成，未保存任何渲染图片；不是重跑在线 SLAM，也不是完整 Adam／RNG 状态恢复。', '',
    '## 结论', '',
    f"联合方案相对匹配基线，训练六档的局部 PSNR 平均变化 **{gates['joint']['view_train_mean_delta_psnr']:+.4f} dB**；五档插值 ratio 的局部平均变化 **{gates['joint']['view_unseen_mean_delta_psnr']:+.4f} dB**。", '',
    f"全局训练六档平均变化 **{gates['joint']['global_train_mean_delta_psnr']:+.4f} dB**，最差单档变化 **{gates['joint']['global_train_min_delta_psnr']:+.4f} dB**。按训练前固定的平均不低于−0.10 dB且每档不低于−0.20 dB门槛，结果为 **{'通过' if gates['joint']['passed'] else '未通过'}**。这是单场景单seed筛查，不是多seed统计非劣证明。", '',
    '1%／5%单独结果见下表，不能用高ratio均值掩盖低ratio退化。未按评价结果调整权重、终点或 checkpoint；评价均采用预先确定的11,580步终点。', '',
    '联合方案在全部11档局部评价的PSNR和SSIM上均改善，自然局部预算的实际K在三组间逐帧相同；没有额外重选点来凑same-K。1%／5%／10%及低比例插值的LPIPS改善，20%–70%的LPIPS小幅变差，不能称为所有质量指标同向提升。', '',
    '1%全局PSNR下降0.1893 dB，距离训练档单项筛查下限仅约0.0107 dB；未见20%全局插值档下降0.2062 dB，Full下降0.0813 dB。预设门槛只定义于训练六档，不能把通过训练档筛查写成所有ratio都保持质量。因此结论是局部监督在本场景有效，但低ratio全局及部分插值质量仍有代价，值得继续验证，暂不据此替换生产默认。Full本身略降，排除了“局部收益只是Full整体PSNR提高”的简单解释，但尚未隔离Selector、Compensator和Master各自的贡献。', '',
    '## 配对设置', '',
    '- 起点为 fr1_xyz RGB-D `seed0_repeat2/11980_shutdown/ply` 的未训练LoD Full导出，Gaussian=13,298，Anchor=2,790，训练关键帧72。',
    '- 三组都是fresh Adam／RNG、同一Master／Selector／Feature初值；完整11,580步相机序列一致，每档ratio 1,930步。',
    '- 训练ratio为0.01／0.05／0.1／0.3／0.5／0.7；仅评价的插值ratio为0.03／0.075／0.2／0.4／0.6。',
    '- 原基线为Full＋全局LoD；联合方案为Full＋全局LoD＋局部LoD，外层1:1:1；额外全局对照将第三分支候选改为全图，第三分支仍只有Photo＋Budget，无额外GI。',
    '- 局部训练使用可见名义Anchor的全部成员，原Gumbel-ST＋预算损失，不做Top-K；推理使用原Viewer同口径候选及自然严格K，不额外进行same-K重选。',
    '- Master允许更新，Fine期间拓扑固定；相机轨迹不变，因此ATE不用于区分这三组。', '',
    '## Full质量', '', '| 组 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |', '|---|---:|---:|---:|']
for group in labels:
    lines.append(f"| {labels[group]} | {value(group,'full',None):.4f} | {value(group,'full',None,'ssim'):.5f} | {value(group,'full',None,'lpips'):.5f} |")
for scope, title in [('global', '全局ratio导出'), ('view', '局部视锥预算推理')]:
    lines += ['', '## '+title, '',
        'Δ均为联合方案减原基线；SSIM越高越好，LPIPS越低越好。标“插值”的ratio未参与训练。K为联合方案在评价帧上的实际选中数量均值。', '',
        '| ratio | 原基线PSNR | 联合PSNR | ΔPSNR | 额外全局PSNR | ΔSSIM | ΔLPIPS | 联合实际K |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for ratio in ALL:
        delta = value('joint', scope, ratio)-value('baseline', scope, ratio)
        ds = value('joint', scope, ratio, 'ssim')-value('baseline', scope, ratio, 'ssim')
        dl = value('joint', scope, ratio, 'lpips')-value('baseline', scope, ratio, 'lpips')
        label = f'{100*ratio:g}%' + ('（插值）' if ratio not in TRAIN else '')
        lines.append(f"| {label} | {value('baseline',scope,ratio):.4f} | {value('joint',scope,ratio):.4f} | {delta:+.4f} | {value('global_control',scope,ratio):.4f} | {ds:+.5f} | {dl:+.5f} | {value('joint',scope,ratio,'K'):.2f} |")
lines += ['', '## 局部逐帧与Bank／非Bank', '',
    '| ratio | ΔPSNR中位数 | ΔPSNR p10／p90 | 改善视角比例 | Bank ΔPSNR | 非Bank ΔPSNR | 实际K相同视角比例 |',
    '|---|---:|---:|---:|---:|---:|---:|']
for ratio in [.01, .05, .1, .3, .4, .5, .7]:
    paired_row = next(r for r in paired if r['group'] == 'joint' and r['scope'] == 'view' and abs(float(r['ratio'])-ratio) < 1e-9 and r['partition'] == 'all')
    bank_delta = value('joint','view',ratio,partition='bank')-value('baseline','view',ratio,partition='bank')
    nonbank_delta = value('joint','view',ratio,partition='nonbank')-value('baseline','view',ratio,partition='nonbank')
    lines.append(f"| {100*ratio:g}% | {paired_row['delta_psnr_median']:+.4f} | {paired_row['delta_psnr_p10']:+.4f}／{paired_row['delta_psnr_p90']:+.4f} | {100*paired_row['psnr_improved_fraction']:.1f}% | {bank_delta:+.4f} | {nonbank_delta:+.4f} | {100*paired_row['same_K_fraction']:.1f}% |")
lines += ['', '## 额外全局对照的解释', '',
    f"额外全局对照相对原基线，局部训练六档平均ΔPSNR为{gates['global_control']['view_train_mean_delta_psnr']:+.4f} dB，全局六档平均为{gates['global_control']['global_train_mean_delta_psnr']:+.4f} dB。联合相对该对照的局部均值差为{gates['joint']['view_train_mean_delta_psnr']-gates['global_control']['view_train_mean_delta_psnr']:+.4f} dB。", '',
    '对照的两路采样器状态隔离，但都从seed0、相同全图人口开始，实际K在11,580步中逐步一致。因此该组主要控制增加一次全局Photo／Budget监督与渲染，不能声称提供了独立全局采样多样性，或与局部工作量完全相等。共同随机数的相关性与候选点量差异均是控制实验的限制。', '',
    '## 实际训练预算与耗时', '',
    '| 组 | Full／全局／第三分支渲染次数 | Adam次数 | 全局累计选中点数 | 第三分支累计选中点数 | 平均Mapping ms | 进程秒 |',
    '|---|---:|---:|---:|---:|---:|---:|']
for row in audit['groups']:
    name = row['group']
    lines.append(f"| {labels[name]} | {row['full_renders']}／{row['global_renders']}／{row['extra_renders']} | {row['steps']} | {row['global_selected_total']:,} | {row.get('extra_selected_total',0):,} | {row['mean_mapping_ms']:.3f} | {status['groups'][name]['seconds']:.2f} |")
lines += ['', '每组另有864次全训练视角GI刷新渲染，未混入训练Mapping次数。六张Feature表Adam时钟均为11,580／9,650／7,720／5,790／3,860／1,930。局部分支使训练曝光及总LoD权重增加；相同Adam次数不意味着相同计算量。GPU存在其他任务，进程秒还包含启动和轮询误差，时间差只作运行记录，不作为严格性能增幅。', '',
    '## 联合方案末2,000步局部预算', '',
    '| ratio | 平均候选N | 平均目标rN | 平均实际K | 平均绝对相对预算误差 |',
    '|---|---:|---:|---:|---:|']
for row in audit['groups'][1]['late_ratio']:
    lines.append(f"| {100*row['ratio']:g}% | {row['extra_candidate_N_mean']:.2f} | {row['extra_target_K_mean']:.2f} | {row['extra_actual_K_mean']:.2f} | {100*row['extra_budget_relative_error_mean']:.2f}% |")
lines += ['', '局部训练预算是软约束，5%–50%存在系统性超选，30%的平均绝对相对误差约18.2%。实际K不能写成目标K。严格K推理会重新截取集合，这一训练／推理差异仍存在；末2,000步未发生预算不可达或有候选零选点，完整训练未触发健康停止。', '',
    '## 指标与证据边界', '',
    '- 全轨迹792共同RGB帧，以原保存相机轨迹匹配真值；Bank／非Bank按实际训练关键帧文件划分，不将非Bank称作独立数据集。',
    '- FP32 RGB在[0,1]截取后计算逐帧PSNR、TorchMetrics 11×11 Gaussian SSIM、AlexNet LPIPS（normalize=True）和L1；统一设置关闭TF32，评价批大小8，三组协议相同。',
    f'- 共{len(raw):,}条逐帧结果保存在PER_VIEW.csv；PAIRED_DELTAS.csv保留每帧差值。p10／p90描述视角分布，不是独立样本置信区间。',
    '- 以原生Rasterizer贡献计数绑定一次性获取颜色、radii和贡献点数；首／中／末帧逐case与正常渲染颜色、radii逐位核对。',
    '- GEOMETRY_SAMPLES.json包含首／中／末帧的位移Anchor单位、Scale倍数和旋转角p50／p90／p99／max，属于有限视角诊断，不冒充全轨迹几何统计。',
    '- Master、Selector和Compensator共同更新，Full质量也可能改变。跨模型质量变化不能全归因于Selector排序；本轮没有固定Master的因果隔离。',
    '- 仅单场景、单seed，未依据非Bank指标挑选终点或调参；生产默认保持。', '',
    '## 后续最有信息价值的验证', '',
    '优先对原基线和联合方案做配对seed复验，确认1%全局−0.189 dB是否稳定、是否跨过既定−0.20 dB门槛。若继续改善当前方案，可单独比较局部外层权重1与0.5，保持相机、步数、网络和其他Loss不变，检验局部收益与低ratio全局代价是否来自三路梯度权重；这是后续独立对照，本轮没有事后调权重。', '',
    '另需固定Master与两网络，比较Gumbel采样的实际局部K和严格预算截取后的质量，针对末期超选定位训练／推理集合差异。该诊断只能解释固定模型，不冒充新的完整训练结果。', '',
    '## 文件位置', '',
    '- 联合模型：`runs/full_fine_joint/23560_shutdown/ply`。',
    '- 基线模型：`runs/full_fine_baseline/23560_shutdown/ply`。',
    '- 对照模型：`runs/full_fine_global_control/23560_shutdown/ply`。',
    '- 服务器根目录：`/home/prmi/zwj/experiments/global_view_joint_lod_20261105`。',
    '- 详细指标：`evaluation/reports/SUMMARY.csv`、`PER_VIEW.csv`、`PAIRED_SUMMARY.csv`。',
    '- 核验：`reports/FULL_TRAINING_AUDIT.json`、`evaluation/reports/INPUT_INTEGRITY.json`、`PREDECLARED_QUALITY_GATES.json`。',
    '', '## 主项目完整性', '',
    '完整训练和评价后，再次核查 `/home/prmi/zwj/mycartgs` 与 `/home/prmi/zwj/realLOD` 的604个受保护文件，全部SHA256一致；508个已有研究快照文件也保持一致，渲染图片输出为0。结果见 `reports/FULL_EXPERIMENT_INTEGRITY.json`。训练、评价和模型均在独立研究目录，未commit／push，未修改生产默认。',
    '', '## 曲线图', '',
    '![全局和局部PSNR及实际点数](figures/PSNR_GLOBAL_VIEW.png)', '',
    '![PSNR、SSIM和LPIPS差值](figures/METRIC_DELTAS.png)',
]
(REPORTS/'FR1_XYZ_GLOBAL_VIEW_JOINT_RESULT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(json.dumps(gates, indent=2))
