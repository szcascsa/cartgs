# 固定十点显式 Anchor 训练诊断

本轮只做源码审查、原 checkpoint 只读测量和带生命周期记录的隔离复跑；未修改本地 production 算法、阈值、Loss、Selector 或 pruning 规则。未 stage、commit、push。

## 结论与证据范围

已经证实的主要问题是：固定十点 + 全部失败才整组删除，保留了大量逐点已失败且几乎不贡献渲染的成员；这些点还占据部分 LoD 选择预算。初始化支持稀疏、空间 key 重建和短寿命情况见下面的独立复跑。它们是结构与训练行为诊断，不能直接证明此前 Full 质量退化的因果来源。

- **原训练**：fr1_xyz，seed 2026/2027，10,000 Mapping；保留原模型与原日志。
- **诊断复跑**：相同场景、seed、配置、10,000 Mapping，仅增加出生/删除/逐点 predicate 日志。在线 SLAM/Mapping 异步，插桩会影响到达时序，不能把复跑历史写成原运行的精确历史。
- 本地、服务器 production `/home/prmi/zwj/mycartgs`、原隔离训练 source 的 29 个受控文件 SHA 全部相同。原 source 与原模型在诊断前后 SHA 均未改变。
- 原 checkpoint 没保存 birth source/count、逐 UID death step 或 deleted-key 历史；`exist_since_iter_` 也未导出到 PLY。因此原运行 Q1/Q3/Q4 的精确答案不可追溯。
- 轨迹差异明确可见：seed2026 原/复跑最终 Anchor 为 2876/3016（+4.87%）；seed2027 为 3019/3392（+12.36%）。因此历史统计仅支持机制诊断，不是原训练逐事件重现，也不能用复跑结果代替原质量。

## 1. 初始化来源

“真实观测数”定义为出生时传入 `observeBlocks()` 的当前批次不同 XYZ 数，不是像素观测次数，也不是该区域最终累计观测数。`growBlocks()` 的输入来自已优化 Gaussian，单独分类，不能称为真实 SLAM 观测。

|诊断复跑 seed|最终 Anchor|出生于真实观测|其中仅 1–3 真实观测|占最终全部|占观测来源|Gaussian 增长来源|
|---|---:|---:|---:|---:|---:|---:|
|2026|3016|2317|1739|57.66%|75.05%|699|
|2027|3392|2619|2003|59.05%|76.48%|773|

1/2/3 个观测的最终存活 Anchor 分解：

- seed 2026：{'1': 1073, '2': 429, '3': 237}；按 L0/L1/L2：{'0': 520, '1': 1219, '2': 0}。
- seed 2027：{'1': 1198, '2': 524, '3': 281}；按 L0/L1/L2：{'0': 638, '1': 1365, '2': 0}。

源码行为：少于 10 个不同来源时，重复使用来源点并对 **全部 10 个 slot** 作确定性小扰动（半径 0.025h，cell 内约束）；因此 1–3 个观测至少缺少 7–9 个独立真实位置支持。它不是复制十份完全相同 XYZ，但也不是十个独立真实观测。出生时只看当前 batch，后续已占用 cell 的观测不会补充 slot 初始化支持；图像重建优化仍会继续。

最终组内低 opacity 比例与出生支持的关系（复跑，相关性不代表因果）：

|seed|出生来源|存活 Anchor|低 opacity 成员比例|
|---|---|---:|---:|
|2026|real_obs_1_3|1739|64.28%|
|2026|real_obs_4_plus|578|60.21%|
|2026|growth|699|56.98%|
|2027|real_obs_1_3|2003|62.21%|
|2027|real_obs_4_plus|616|59.20%|
|2027|growth|773|54.49%|

## 2. Pruning 保留情况：原模型实测

正式配置 `densify_min_opacity=0.02`、`prune_big_point_after_iter=-1`；因此本次逐点 predicate 是 `opacity < 0.02`，没有 screen/world size 删除条件。以下是最终状态 predicate，不能冒充任意过去 pruning 时刻的 predicate。

|seed|存活 Anchor / Gaussian|含失败成员的 Anchor|至少 5 成员失败|逐点失败成员 / 比例|失败点平均 opacity|失败点渲染权重占比|
|---|---:|---:|---:|---:|---:|---:|
|2026|2876 / 28760|2660|2228|18094 / 62.91%|0.000655912|0.0315%|
|2027|3019 / 30190|2795|2261|18412 / 60.99%|0.0007903|0.0136%|

|失败成员数 / 10|seed2026 Anchor|seed2027 Anchor|
|---:|---:|---:|
|0|216|224|
|1|70|86|
|2|95|111|
|3|117|152|
|4|150|185|
|5|223|241|
|6|298|347|
|7|406|417|
|8|571|602|
|9|730|654|
|10|0|0|

每个 Anchor 的十个 opacity、失败成员数、总/失败成员渲染权重、渲染非零成员数都保存在 `frozen/original_surviving_anchors_with_contribution.csv`；逐 Gaussian 的 UID、slot、opacity、累计权重保存在 `frozen/gaussian_contributions.csv`。

**真实贡献测量**：从原先 792 个共同视图等间隔预选 64 个。在原渲染器中，以预计算颜色的 VJP 测量 `d(sum image_R)/d(color_i,R) = sum_pixels alpha_i*T_i`。颜色不影响 alpha/深度顺序；没有 image clamp，没有 optimizer step。测量前后原模型 SHA 不变，权重和图像总 alpha 的守恒误差 < 2e-4（实际约 1e-7）。这不是旧 SI 的 `opacity*T` 代理量，也不是质量增益归因指标。

- seed 2026：失败点中 95.84% 在这 64 视图累计贡献为零；累计权重总占比 0.0315%。
- seed 2027：失败点中 94.81% 在这 64 视图累计贡献为零；累计权重总占比 0.0136%。

临时移除所有 opacity<0.02 成员的只读敏感性测试（同 64 视图，未重训）：

|seed|Full → 临时移除后点数|ΔPSNR (dB)|ΔSSIM|ΔLPIPS|
|---|---:|---:|---:|---:|
|2026|28760 → 10666|-0.004474|+0.0000109|+0.0000969|
|2027|30190 → 11778|-0.000296|-0.0000045|+0.0000330|

这是删点敏感性实验，**不是 same-K 方法对比，也不是已获准的生产 pruning 修改**；渲染贡献不等同独立质量贡献。

### 额外暴露的问题：不可渲染成员仍占预算

`forward.cu` 定义 `alpha=min(0.99, opacity*exp(power))`，`power<=0`，并跳过 `alpha<1/255`。因此在当前 opacity 不变时，`opacity<1/255` 的成员对任何视图都无法贡献像素；TF 目前仅改 XYZ/scale/rotation，不改 opacity。这些点没有 photo 渲染梯度，Full opacity regularizer 仍可施加向下压力；不能据此断言所有后续优化中 opacity 永远不变。

|seed|策略|ratio|实际选择|低于 pruning 阈值|低于 renderer 截断|截断成员比例|
|---|---|---:|---:|---:|---:|---:|
|2026|Full|100%|28760|18094|17330|60.26%|
|2026|cached_bernoulli|1%|305|2|1|0.33%|
|2026|cached_bernoulli|5%|1434|158|133|9.27%|
|2026|cached_bernoulli|10%|2868|809|730|25.45%|
|2026|cached_bernoulli|15%|4363|1897|1768|40.52%|
|2026|same_k|1%|256|2|1|0.39%|
|2026|same_k|5%|1283|138|117|9.12%|
|2026|same_k|10%|2567|665|595|23.18%|
|2026|same_k|15%|3851|1505|1403|36.43%|
|2027|Full|100%|30190|18412|17450|57.80%|
|2027|cached_bernoulli|1%|297|2|1|0.34%|
|2027|cached_bernoulli|5%|1485|258|234|15.76%|
|2027|cached_bernoulli|10%|3035|1073|930|30.64%|
|2027|cached_bernoulli|15%|4617|2180|1981|42.91%|
|2027|same_k|1%|256|2|1|0.39%|
|2027|same_k|5%|1283|244|222|17.30%|
|2027|same_k|10%|2567|868|769|29.96%|
|2027|same_k|15%|3851|1565|1391|36.12%|

## 3. 同一空间 key 的重复创建

原模型最终同层空间碰撞桶为零，只能说明最终快照没有同时重叠的 block，不能说明历史没有重建。UID 是新 block 的永久身份；空间身份是 `(level,kx,ky,kz)`。本次复跑没有几何/CD 迁移事件，因此下面的同 key 统计不混入几何运输。

|复跑 seed|总创建|不同空间 key|创建过多次的 key|额外重建事件|单 key 最大创建次数|同时重复创建|
|---|---:|---:|---:|---:|---:|---:|
|2026|12290|6914|2612|5376|12|0|
|2027|13540|7742|2704|5798|12|0|

同 key 创建次数分布：

- seed 2026：{'1': 4302, '2': 1296, '3': 660, '4': 289, '5': 158, '6': 97, '7': 55, '8': 26, '9': 20, '10': 7, '11': 3, '12': 1}。
- seed 2027：{'1': 5038, '2': 1343, '3': 618, '4': 316, '5': 155, '6': 120, '7': 65, '8': 46, '9': 33, '10': 6, '11': 1, '12': 1}。

源码原因：`pruneBlocks()` 删除 block 后，`rebuildBlockLookup()` 只保留当前存活 key，没有删除墓碑/历史；以后新观测或 gradient growing 再次命中同 key，就会新建完整十点并分配新 UID。这是删除后重建，不是 UID 重复、也不是已占用 key 去重失效。逐 key 全部 UID/来源在 `replay_spatial_key_creation_counts.csv`，相邻两次创建的空窗在 `replay_recreation_events.csv`。

## 4. Anchor 寿命

时间单位是 Mapper 已完成 Mapping 计数（`protection.mapping_iteration`，初始建图记 0）；不是视频帧、不是 Gaussian optimizer 更新次数。在线观测批次按上一个已完成 Mapping 计数入账，精度约一个 Mapping 步。删除寿命=`death_mapping-birth_mapping`。快速死亡阈值预先固定为 ≤1/10/50/100/250/500/1000，不按结果挑阈值。

|复跑 seed|创建 / 删除 / 存活|已删平均寿命|中位数|p10 / p90|已删中 ≤100 步|已删中 ≤500 步|
|---|---:|---:|---:|---:|---:|---:|
|2026|12290 / 9274 / 3016|674.47|444.0|133.0 / 923.3|8.77%|84.54%|
|2027|13540 / 10148 / 3392|676.69|486.0|146.0 / 906.0|5.20%|86.28%|

避免期末右删失偏差：仅对出生≤9500、至少有500步观察机会的 Anchor，统计“出生后500步内删除 / 所有可观察出生”：

- seed 2026：7840 / 12129 = 64.64%。
- seed 2027：8756 / 13317 = 65.75%。

分来源的周转情况：

|seed|来源|创建|删除比例|已删平均寿命|已删中 ≤500 步|
|---|---|---:|---:|---:|---:|
|2026|growth_fine|2677|80.43%|559.2|96.05%|
|2026|growth_mixed|27|74.07%|675.0|80.00%|
|2026|growth_same|682|75.37%|571.0|96.11%|
|2026|initial_observation|1126|25.22%|4480.6|0.00%|
|2026|online_observation|7778|81.04%|550.8|83.48%|
|2027|growth_fine|3231|82.39%|569.3|96.36%|
|2027|growth_mixed|47|80.85%|513.2|97.37%|
|2027|growth_same|831|76.53%|564.5|96.54%|
|2027|initial_observation|1075|21.58%|4435.3|0.00%|
|2027|online_observation|8356|78.75%|599.4|84.19%|

已删除 Anchor 的平均寿命不是所有 Anchor 的预期寿命；存活 Anchor 是右删失样本。pruning 约每500 Mapping执行，寿命会受这个检查节奏量化。细层增长来源组的高短期删除率说明 growing 存在大量周转，但不能单独判断这些增长对质量的净效应。分来源的寿命与所有预注册阈值见 `replay_lifetime_summary.csv`；逐 UID 生死时刻见 `replay_deleted_anchor_lifetimes.csv`。

## 5. 问题定位与后续验证边界

1. **已证实的结构代价**：十点强绑定 + 全失败才删，使大量单独失败的 slot 长期留存。这符合此前确认的规则，不是实现违反固定十点约束。
2. **已证实的选择代价**：低于 renderer 截断的 slot 被 LoD 选中，占用真实预算；same-K 数量匹配也不等于有效贡献数量匹配。
3. **初始化风险**：少量当前批次来源撑起完整十点，且粗细层各初始化一组；它提供几何自由度，也增加弱支持成员。复跑支持数与最终低 opacity 的相关统计不足以证明初始化是唯一原因。
4. **生命周期风险**：删除 lookup 历史后允许同 key 再生，可能形成创建—优化—整组删—再创建的周转；实际次数与短寿命比例已量化。增长候选和真实观测必须分开分析。
5. **可观测性缺口**：原训练缺逐 UID 事件，无法追溯其三项历史问题。独立诊断补齐记录，但没有把复跑的数值冒充原运行。
6. 当前证据不支持把问题归因到“MLP 不够大/训练步数不够”；它首先暴露的是 slot 有效性、预算利用率、支持来源和生命周期周转。也不能用这次诊断宣称已经解释所有 Full/LPIPS 差异。

本轮没有更改固定十点、整组 prune、增长阈值、网络、Loss、TF、CD 或训练轮转。任何后续修复都应单独对照，并同时保留真实 N、有效贡献 N、same-K、Full 与 LoD 指标。

## 6. 验证与产物

- 原模型只读：opacity 全量统计；64共同视图 alpha*T VJP 与守恒检查；相同视图的低 opacity 删除敏感性；原导出 mask 的预算占用。
- 复跑：CPU/CUDA 显式十点结构测试；CPU fixture 与原版本 exact；CUDA fixture 输出 exact，仅保存的 Anchor Feature 存在最大 1.17e-10 差异，allclose(atol=2e-6, rtol=2e-5) 通过。补做原二进制 CUDA repeat，其与原 fixture 的差异为0；没有把 CUDA 参数 bitwise gate 写成通过。两个10,000 Mapping复跑的出生UID唯一、出生减删除等于最终UID集合、N=10M、无同时重复key、寿命非负、日志计数闭合。
- 源码与模型保护：29文件 SHA、本地 HEAD/index 核对、原 checkpoint SHA。诊断只保存在独立目录。
- 服务器原实验：`/home/prmi/zwj/octree-explicit-20261021`。诊断：`/home/prmi/zwj/octree-explicit-diagnostics-20261021`。
- 本地明细：`tests/diagnostics/explicit_anchor_lifecycle_20261021/`。其中 `frozen/` 是原模型测量，`replay/` 是带事件记录的独立复跑。

![原模型与复跑诊断](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/explicit_anchor_lifecycle_20261021/diagnosis.png)

源码依据：[初始化与补点](<H:/AIProjects/cartgs优化更细致版/mycartgs/improvements/selector/explicit_anchor_blocks.h:39>)；[整组删除与 lookup 重建](<H:/AIProjects/cartgs优化更细致版/mycartgs/improvements/selector/explicit_anchor_blocks.h:193>)；[逐点 pruning predicate](<H:/AIProjects/cartgs优化更细致版/mycartgs/src/gaussian_model.cpp:923>)；[Full opacity regularizer](<H:/AIProjects/cartgs优化更细致版/mycartgs/src/gaussian_mapper.cpp:1589>)；[renderer alpha cutoff](<H:/AIProjects/cartgs优化更细致版/mycartgs/cuda_rasterizer/forward.cu:411>)；[TF 属性输出](<H:/AIProjects/cartgs优化更细致版/mycartgs/python/transform_field.py:188>)。
