"""生成原版 Anchor pruning 与逐 slot 窗口诊断的中文交付报告。"""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

here=Path(__file__).resolve().parent;repo=here.parents[2]
e=here/'evidence';v=e/'validation'
s=json.loads((v/'audit_summary.json').read_text())
gates=json.loads((v/'correctness.json').read_text())
verification=json.loads((e/'delivery_verification.json').read_text())
assert len(gates)==11 and all(x['passed'] for x in gates.values())
with (v/'final_anchor_diagnostics.csv').open() as f:anchors=list(csv.DictReader(f))
with (v/'final_slot_diagnostics.csv').open() as f:slots=list(csv.DictReader(f))
with (v/'window_summary.csv').open() as f:windows=list(csv.DictReader(f))
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False
fig,ax=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True)
x=np.arange(4);w=.34
ax[0].bar(x-w/2,[r['current_cutoff_budget_fraction']*100 for r in s['lod_budget']],w,label='当前低于截断')
ax[0].bar(x+w/2,[r['history_and_current_cutoff_selected']/r['selected']*100 for r in s['lod_budget']],w,label='充分窗口始终低于截断且当前仍低')
ax[0].set(xticks=x,xticklabels=['1%','5%','10%','15%'],ylabel='占实际选择预算 (%)',title='原 Cached Bernoulli：未屏蔽候选')
ax[0].legend(fontsize=8)
known=[r for r in anchors if r['sufficient']=='True']
hist=np.bincount([int(r['always_below_cutoff_slots']) for r in known],minlength=11)
ax[1].bar(np.arange(11),hist,color='C1')
ax[1].set(xlabel='充分窗口内始终低于 1/255 的 slot 数 / 10',ylabel='最终存活 Anchor 数',title='具有充分窗口记录的存活 Anchor')
fig.suptitle('fr1_xyz / seed2026 / 10,000 Mapping：仅作诊断')
fig.savefig(here/'pruning_diagnostics.png',dpi=180);plt.close(fig)
example=s['example'];uid=example['uid'];sample=[r for r in slots if int(r['uid'])==uid]
budget=s['lod_budget']
text=f'''# Octree-GS 原版 Anchor pruning 与逐 slot 窗口统计

## 结论

已按要求先提交修改前基线：`497463a664fa198dd1bec0be2f3124bd7f878899`，分支 `redundancy-aware-si`，未 push。本轮 pruning 与诊断实现保留为未提交改动。

**原版整组删除规则与逐 slot 同窗口统计均已实现并通过验证。长期低 opacity 成员及 LoD 预算浪费仍然严重，但本轮没有加入候选屏蔽或任何额外删除条件。**

fr1_xyz、seed2026、10,000 Mapping 的一次诊断训练：创建 {s['created_anchors']} 个 Anchor，整组删除 {s['pruned_anchors']} 个，最终 {s['final_anchors']} Anchor / {s['final_gaussians']} Gaussian。最终 {s['final_current_below_cutoff']} 个 Gaussian（{s['final_current_below_cutoff_fraction']:.2%}）当前低于 renderer 截断；{s['final_current_below_002']} 个（{s['final_current_below_002_fraction']:.2%}）低于旧逐点阈值。

其中 {s['final_with_sufficient_history']} 个最终 Anchor 有充分窗口记录，{s['final_without_sufficient_history']} 个没有充分记录；后者不会被统计成“长期失效”。充分记录者中 {s['final_sufficient_anchor_with_ge8_always_cutoff']} 个至少有 8/10 slot 全窗口始终低于 `1/255`，占充分记录者 {s['final_sufficient_anchor_with_ge8_always_cutoff']/s['final_with_sufficient_history']:.2%}。还有 {s['all_ten_always_cutoff_retained']} 个的十个 slot 都全窗口低于截断，但 Anchor opacity 总和仍达到原版保留门槛。

这是单场景、单 seed 的机制与预算诊断。没有进行配对质量实验，不能据此宣称 PSNR/SSIM/LPIPS 提升或退化。在线异步训练轨迹与旧模型不同，不能把旧/新 N 或预算差异直接解释为纯 pruning 因果效应。

## 1. 原版规则

核对本地官方源码快照 `city-super/Octree-GS@9db64fe8981f5d6a1002c41cbc275a14f4d5ea0c`，四个参考文件的 SHA 与 `SOURCE.json` 一致。原代码：[training_statis](https://github.com/city-super/Octree-GS/blob/9db64fe8981f5d6a1002c41cbc275a14f4d5ea0c/scene/gaussian_model.py#L634)、[adjust_anchor](https://github.com/city-super/Octree-GS/blob/9db64fe8981f5d6a1002c41cbc275a14f4d5ea0c/scene/gaussian_model.py#L852)。

每次可见访问：

`O_A += sum_k max(alpha_Ak, 0)`；`T_A += 1`。

删除严格要求同时满足：

`T_A > check_interval * 0.8`；`O_A < 0.005 * T_A`。

opacity 使用十个 slot 的**总和**，没有除以 10。`>` 与 `<` 均为严格不等式。没有失败成员比例、成员最大值、单点 opacity、screen size 或 world size 的附加删除条件。

达到次数门槛的 Anchor 检查后清零 O/T；不足门槛的继续累计。slot 统计完全同窗重置；不能把固定 500 Mapping 简单当作每个 Anchor 都有 500 次访问。

## 2. 本地适配与梯度边界

- 保留现有 500 Mapping 的拓扑检查调度，`check_interval=densifyInterval()`；因此本次门槛是 **超过 400 次 Full 可见访问**。LoD 轮转中 Full 较少时，统计可以跨多个检查周期累计。
- 官方模型使用 Anchor 可见 mask；本项目有十个显式且可独立移动的 Gaussian。定义“Anchor 可见”为 Full renderer 的 `radius>0` 集合中至少存在一个成员；同一次 Full view 只记一次 Anchor 访问，并统计该 Anchor 全部十个 slot。
- 只在 Full Render、Gaussian optimizer 更新前采集未变形 Master 的激活 opacity。LoD/TF/Change Distill 不累计该统计；Selector 输入和视角无关评分保持原状。
- 所有新统计都在 `NoGradGuard` 下更新并 detach，不进入 optimizer，不产生额外 loss，不进入 Selector 评分或候选 mask。
- 新增 Anchor 的 O/T 与 slot 统计补零；删除使用同一整组 survivor mask 压缩；几何/Loop/CD 迁移保留 UID、slot 归属和其统计。Gaussian 六组属性、Cached Uniform、Anchor Feature 及双 Adam owner 继续同步迁移。
- Legacy 非显式 Anchor 路径保持逐点 pruning。此次仅替换显式十点路径的政策。

## 3. 每个 slot 保存什么

同一 Anchor 窗口与分母 T：

|字段|含义|
|---|---|
|`slot_opacity_accum`|可见访问时 slot 正 opacity 累计值|
|`slot_mean_opacity`|累计值 / Anchor 访问次数|
|`slot_below_002_count/fraction`|低于 0.02 的次数及访问比例|
|`slot_below_cutoff_count/fraction`|低于 1/255 的次数及访问比例|
|`slot_current_opacity`|导出当时的当前激活 opacity|
|`anchor_visits`、`block_uids`、`levels`、`keys`|分母与稳定身份、空间位置|

未观察到的访问不更新 O 或 T，也不为 slot 补零。T=0 的平均值、比例保存为 NaN，CSV 中为缺失值。它与“T>0 且始终低 opacity”明确区分。

每次检查**重置前**保存 `anchor_pruning/window_<mapping>.pt` 和 `windows.csv`；最终 checkpoint 保存 `point_cloud/iteration_10000/anchor_pruning_state.pt`。因此即使最终时刻刚清零，也能读取上一完整窗口。统计随 checkpoint 导出供复核，本轮没有新增跨进程恢复训练协议。

末期诊断对每个存活 UID 选其最近一个观察充分的闭合窗口。没有充分窗口时标记观察不足或未观察，不把它算作长期失效。“始终低于截断”只指这个已记录窗口的全部可见访问，**不是整个训练生命周期或连续空间时间内所有视图**。

## 4. 实际例子：Anchor 保留，成员失效

Anchor UID **{uid}**，L{example['level']}，窗口检查时 Mapping={example['mapping']}，累计 **{example['visits']} 次**可见访问，平均 slot opacity 总和 **{example['anchor_mean_opacity_sum']:.6f}**，满足原版保留条件。

十个成员中，**{example['always_below_cutoff_slots']} 个在该窗口的每次访问都低于 `1/255`**；另外两个成员占累计 opacity 的 **{example['other_slot_opacity_share']:.4%}**。这是“整体 opacity 达标，但多数成员长期低 opacity”的直接实例；没有因此额外删除这个 Anchor。

|slot|窗口平均 opacity|低于 0.02 比例|低于 1/255 比例|最终当前 opacity|
|---|---:|---:|---:|---:|
'''
for r in sample:
    text+=f"|{r['slot']}|{float(r['mean_opacity']):.7g}|{float(r['below_002_fraction']):.2%}|{float(r['below_cutoff_fraction']):.2%}|{float(r['current_opacity']):.7g}|\n"
text+='''
另外存在“10 slot 全窗口都低于 renderer 截断，但总和仍超过 0.005”的保留 Anchor。这也符合原公式：十个较小正 opacity 的和可能达到 0.005；整体保留并不保证其中某个 slot 能跨过 renderer 截断。

## 5. 当前 LoD 预算利用率

直接使用原 `build_selector_mask`、保存的 Cached Uniform 与同一个 Selector，未屏蔽任何候选，也没有重新拟合模型。TF 目前只改 XYZ/scale/rotation，不改 opacity；因此当前 opacity<1/255 的点在该状态下无法通过 renderer 的 alpha 截断。

|ratio|实际选点数|当前低于 1/255|占选择预算|充分历史窗口始终低于截断且当前仍低|
|---|---:|---:|---:|---:|
'''
for r in budget:
    text+=f"|{r['ratio']:.0%}|{r['selected']}|{r['current_below_cutoff']}|{r['current_cutoff_budget_fraction']:.2%}|{r['history_and_current_cutoff_selected']} / {r['selected']}（{r['history_and_current_cutoff_selected']/r['selected']:.2%}）|\n"
text+=f'''
5%/10%/15% 的当前预算浪费依然明显。上表历史指标仅使用有充分窗口记录的成员；观察不足者没有被默认归入失效。结果支持将“候选屏蔽”作为后续独立研究问题，**本轮尚未实现或验证该方案**。

![窗口与预算诊断]({(here/'pruning_diagnostics.png').as_posix()})

## 6. 验证状态与成本

**已测试**：11 项 gate 通过，覆盖 CPU/CUDA 原版公式、访问次数取等不删、opacity 取等不删、sum 而非 slot mean、时间平均而非当前值、观察不足跨窗口累计、NaN 缺失值、暂时低 opacity 的非100%比例、8个长期低 slot 的有效整体仍保留、slot/Anchor 同窗重置、追加与整组压缩、双 Adam owner、几何运输；另有生产 Mapping 的 Full-only 统计、joint/Full 路由、Selector+TF CD、Legacy 回归及 Python/native 兼容测试。

**已运行**：fr1_xyz / seed2026 / 10,000 Mapping，一次诊断训练，{s['checks']} 次 pruning 检查。每个窗口独立复算删除公式、slot sum=Anchor sum、分母与比例、未观察为 NaN、最终 UID 与人口闭合。候选屏蔽未实现。

**未测试**：多场景/多 seed 完整质量矩阵、same-K PSNR/SSIM/LPIPS 对照、严格的无干扰耗时基准。此次训练时长仅为 smoke 运行记录，不能用于声称更快。

统计驻留存储为每 Anchor 256 字节（两个 Anchor 累计量及三组十 slot 累计量），最终约 {s['final_anchors']*256/1024/1024:.2f} MiB；临时张量和定期写文件另计。每次 Full 有额外统计 kernel，检查窗口会同步导出诊断数据；没有将该诊断版本称为已完成低延迟性能验收。

## 7. 文件与源码

服务器隔离验证目录：`/home/prmi/zwj/octree-anchor-pruning-20261021`。原 production、原训练 source 和模型 SHA 已核对保持不变；新原生源码与本地上传的 8 个受控路径 SHA 对齐。

本地证据目录：[anchor_opacity_pruning_validation](<{here.as_posix()}>)。

- [逐 Anchor 窗口明细](<{(v/'anchor_window_statistics.csv').as_posix()}>)、[每个窗口汇总](<{(v/'window_summary.csv').as_posix()}>)。
- [最终存活 Anchor 诊断](<{(v/'final_anchor_diagnostics.csv').as_posix()}>)、[逐 slot 四项统计](<{(v/'final_slot_diagnostics.csv').as_posix()}>)。
- [LoD 预算明细](<{(v/'lod_budget_statistics.csv').as_posix()}>)、[correctness](<{(v/'correctness.json').as_posix()}>)、[汇总](<{(v/'audit_summary.json').as_posix()}>)。
- 原始窗口 PT、最终统计 PT、日志和 SHA 清单均在 `evidence/` 下保留。

主要实现：[统计与删除](<{(repo/'src/gaussian_model.cpp').as_posix()}:130>)、[Full 接入](<{(repo/'src/gaussian_mapper.cpp').as_posix()}:1575>)、[边界测试](<{(repo/'tests/anchor_selector/anchor_opacity_pruning_test.cpp').as_posix()}:1>)。
'''
path=repo/'OCTREE_ANCHOR_OPACITY_PRUNING_REPORT.md';path.write_text(text,encoding='utf-8')
print(path)
