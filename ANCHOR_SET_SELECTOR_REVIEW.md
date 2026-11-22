# Anchor 集合 Selector 审阅

审阅日期：2026-11-02。范围为 `anchor_set_residual48.yaml` 实际启用的集合 Selector、Anchor 名义格子视锥候选、FlexGS、Maturity V2 与 Joint 训练。按用户要求排除未启用的 ChangeDistill。

本文记录修正前的schema 1发现，保留为历史审阅证据。后续按用户授权实现了名义格子相对位置与slot-eligible聚合的schema 2；当前改动和验证见`ANCHOR_SET_SELECTOR_NOMINAL_ELIGIBLE_REPORT.md`，不要将本文风险描述直接视为修正后代码状态。

## 结论

当前实现的维度、成员映射、候选恢复、模型保存加载和 Maturity 私有梯度路径未发现新的硬错误。集合结构能够表达变长成员集合，但有两项较重要的表示/稳定性风险，以及一项数值稳健性问题；不建议仅凭已有小规模正确性测试认定真实在线训练已经验证。

本次没有修改网络、Loss、生产配置或服务器运行库。只增加独立诊断脚本、证据与本报告；没有恢复暂停队列，没有 commit/push。

## 1. 单成员 Anchor 丢失全部局部位置信息

位置：`improvements/selector/anchor_set_encoder.h:26–27`，对应 Python 实现同样如此。

当前使用：

\[
\mu_a=\operatorname{Mean}_{j\in A_a}x_j,\qquad d_i=(x_i-\mu_a)/h_a.
\]

当 Anchor 只有一个成员时，`μ_a=x_i`，所以 `d_i=0`。网络仍然能利用该成员的 Scale、SH DC、Opacity 和 ratio，但不能区分其在名义格子内的位置。对整个成员集合平移时，集合内部几何表示也不变；这是当前设计的平移不变性，同时意味着 Anchor 质心相对格子中心的位置没有输入网络。

本轮诊断：两个位置相差很大的单成员 Anchor，非位置属性相同，Selector 的 XYZ 输入梯度全部为零。分开执行相同批量形状时 logits 逐位相同；合并批量时 logits 最大差 `7.45e-9`，不能把这一级浮点舍入当作位置信息。

读取既有 fr1_xyz 地图的真实 `anchors.member_counts`，结果如下。这些是旧网络地图的占用统计，不是新集合网络的在线质量结果。

| 旧地图 | N | Anchor 数 | 单成员 Anchor 数 | 单成员占 Gaussian 比例 |
|---|---:|---:|---:|---:|
| prune0 / RGB-D / natural | 10,402 | 4,052 | 2,274 | 21.86% |
| prune0 / RGB-D / aligned | 12,125 | 4,520 | 2,552 | 21.05% |
| prune0 / Mono / natural | 10,671 | 4,470 | 2,781 | 26.06% |
| prune0 / Mono / aligned | 11,502 | 4,408 | 2,697 | 23.45% |
| prune1 / RGB-D / natural | 15,320 | 6,369 | 3,527 | 23.02% |
| prune1 / RGB-D / aligned | 14,279 | 4,483 | 1,990 | 13.94% |
| prune1 / Mono / natural | 14,817 | 7,207 | 4,881 | 32.94% |
| prune1 / Mono / aligned | 13,779 | 5,376 | 3,138 | 22.77% |

建议进行下面两种最小修正中的一种，保持集合聚合主体：

| 方案 | 好处 | 代价与取舍 |
|---|---|---|
| 推荐：Encoder 内部改用 `(x_i-c_a)/h_a`，其中 `c_a` 为名义格子中心 | 单成员仍有几何位置；Mean/Max 仍表达整组分布；输入维度和参数量不增加 | 不再对成员集合相对固定格子中心的平移保持不变；不能直接视为原函数 |
| 保留当前质心相对坐标，在 Anchor context 增加 `(μ_a-c_a)/h_a` 三维 | 同时保留内部形状与集合在格子中的位置 | 评分输入 50→53D，增加 96 个共享参数，需要新的 schema 与匹配对照 |

这两种方案都无需把所有成员 offset 平铺拼接，仍支持不定成员数。本次未实施结构修改。是否改善低 ratio 空间覆盖，需要真实在线消融，当前没有 PSNR 结论。

## 2. 不可选成员仍能通过上下文影响可选成员

位置：`anchor_set_encoder.h:24–31`；调用处 `src/gaussian_mapper.cpp:3553–3569`。当前聚合完整 Anchor 成员，slot eligibility 仅用于后续采样，不用于质心、Mean、Max。

构造合法共享网络权重，使某个 Max 通道读取 Scale；第二个成员的 Opacity 为 `1e-6`，通过真实 `mask_inactive_slots` 得到 `[true,false]`。仅改变该不可选成员的 Scale，可选成员的 keep 概率变为：

| 不可选成员 Scale | 可选成员 keep logit | 可选成员 keep 概率 |
|---:|---:|---:|
| 0.1 | 0.01 | 0.5025 |
| 1 | 0.1 | 0.5250 |
| 10 | 1 | 0.7311 |
| 1,000 | 100 | 1.0000 |

这是控制权重反例，证明影响通路存在；没有证明训练好的网络一定学到了这组权重，也没有证明已经发生塌缩。低 Opacity 本身不是质量差的证据，问题是这些成员已被现有 slot 规则判为不可选，仍能主导邻居上下文。等权 Mean 和质心也存在相同的数据资格不一致问题。

建议对照：聚合前使用现有 slot eligibility，质心、Mean、Max 与 count 使用一致有效成员集合，并明确处理全无效 Anchor。不要额外把所有低 Opacity 成员都排除；只复用现有可逆 slot 规则。训练和推理必须同时修改。此项改变集合定义，需要单独消融，本次未实施。

## 3. 世界坐标直接求均值存在可避免的精度损失

位置：`anchor_set_encoder.h:26–27`。

当前先对世界坐标求和、除以成员数，再用世界坐标减去质心。世界坐标远大于局部尺寸时，FP32 的中间舍入会放大到归一化局部坐标。使用同一份已存 FP32 坐标的 FP64 结果作为参照，排除了输入坐标量化本身的影响；`h=0.1` 时实测：

| 世界平移量 | 当前归一化坐标最大绝对误差 | 先减局部参考再求均值 |
|---:|---:|---:|
| 0 | 1.68e-8 | 1.68e-8 |
| 1 | 3.97e-7 | 9.93e-9 |
| 10 | 3.18e-6 | 9.93e-9 |
| 100 | 5.09e-5 | 9.93e-9 |
| 1,000 | 2.03e-4 | 4.97e-9 |
| 10,000 | 6.51e-3 | 9.93e-9 |

大平移属于压力测试，不代表当前 TUM 场景的坐标范围或质量损失。若保留质心相对表示，建议先计算 `q_i=x_i-c_a`，再计算 `q_i-Mean(q)`；数学上等价，FP32 不保证逐位相同，因此也需要记录差异。若采用第 1 项推荐的名义格子相对坐标，该次世界坐标均值计算可直接消除。

## 4. Joint 输入梯度属于有意引入的耦合，需要观察

当前 Photo/Budget 可以通过集合 Encoder 更新 Master 输入。一个成员的评分依赖同 Anchor 其他成员，因此其他成员即使没有直接被渲染，也可能收到评分路径梯度。

仅对 row0 的 keep logit 求 VJP，三个成员 XYZ 梯度 L1 分别为 `0.032114 / 0.015463 / 0.016650`；Scale 梯度 L1 为 `0.006781 / 0.000849 / 0.001584`。这单独证明新增的跨成员路径。原逐点 Selector 中，row0 的 logit 不依赖另外两行输入。

另一个 budget-only 诊断使用全图目标 `0.01×200`，hard mask 为 `[1,0,0]`，未选两行也有非零 Master 梯度。但原 ST Selector 自身也会对未选点产生 Budget 输入梯度，不能把这条现象全部归因于集合聚合。

此处没有发现 Maturity 隔离失效。Maturity 仍只对 `local_emd.* / soft_net.* / selector_set_encoder.*` 等 Selector 私有参数求 VJP。不要直接把输入 detach 当作修复；那会改变当前 Joint 训练语义。后续应记录 Photo 与 Budget 分别对成员几何的梯度范数，再判断是否需要受控隔离对照。

## 5. 已通过的检查与验证边界

已有独立构建与测试证据位于 `research/anchor_set_selector_20261102/evidence/`；本轮诊断位于 `research/anchor_set_selector_review_20261102/`。

| 检查 | 结果 |
|---|---|
| 变长成员、Mean/Max、空集合、成员/Anchor 重排 | 已有 C++ CPU/CUDA 与 Python 测试通过 |
| compact 候选使用全局 owners，恢复全图 Mask | 已有实际 `accumulateLodGradients` 合成地图测试通过；预算目标仍按全图 N |
| Maturity 仅更新 Selector | 已有隔离测试通过；本轮复核私有参数名单 |
| C++ / Python 前向及 VJP | 已有 66 项对照通过，最大绝对误差 `5.96e-8` |
| 新增数值梯度检查 | 本轮 CPU FP64 `gradcheck` 通过：XYZ、Scale、SH DC、Opacity |
| 保存加载、Frozen 克隆、拓扑行映射 | 已有功能测试通过 |
| 原11D采样与Residual48回归 | 已有测试通过；Residual48 97项逐位检查通过 |
| 真实在线训练稳定性、PSNR与完整 Mapping 性能 | 尚未验证，不据合成测试作结论 |

已有合成 Mapping 测试为 `N=16`、候选成员 `C=4`，部分测试人工调整了 soft_net 以确保触发选点或梯度路径，不能替代默认初始化下的连续训练。新的集合模式也会绕过只支持11D的融合 Selector MLP，并增加全局 Anchor 行空间的聚合 workspace；性能尚未测量。

## 6. 完整性与下一步

本轮重新核对服务器主项目 334 个受保护源码、配置、二进制和动态库文件，SHA256 无变化；两批原实验仍 `paused=true`。见 `REVIEW_INTEGRITY.json`。原验证日志与本次初始诊断均保留。

推荐顺序：优先明确上下文是否排除 slot-inactive 成员，并验证推荐的名义格子相对位置编码；分别保留单独对照。之后用默认初始化执行真实 Full 预热→LoD 训练，记录单成员/多成员评分、Max 来源、Scale 长尾与 Photo/Budget 输入梯度。质量和训练稳定性成立后再测算聚合开销及融合空间。本次不启动正式场景训练或恢复历史队列。
