# S00R00E08D12 默认配置接入

后续更新：当前默认已按用户要求提升到E50；本文保留E8接入记录，E8配置为 `cfg/selector/s00_e08_reference.yaml`，E50实测见 `S00_E50_COMPENSATION_AUDIT.md`。

本轮按用户指定，将项目默认配置切换为S00R00E08D12。R3 Independent14D继续作为希望达到的质量参考，独立配置为 `cfg/selector/r3_reference.yaml`。历史矩阵报告中的“保留R3默认”是当时建议，本轮用户决定取代该建议；历史质量数据保持原样。

## 当前默认结构

```text
输入：Local Offset3 + activated Scale3 + SH DC3 + activated Opacity1 + log(ratio)1
共享Encoder：11 → 32 ReLU
Selector私有头：32 → 32 ReLU → 2
补偿输入：共享h32 + 每Anchor/ratio独立latent8
Compensator私有头：40 → 32 ReLU → 32 ReLU → 14
```

Selector不保存可学习Anchor latent。零列的Anchor表只保留拓扑行数，不参与梯度；四档补偿latent分别存储和更新。仍先Cached Bernoulli选点，只对选中K行读取补偿latent、解码14维属性补偿。

未见ratio对相邻训练节点的8维latent作raw-ratio线性插值，再解码；训练范围外latent取最近端点，Encoder的log(ratio)仍使用实际ratio。范围外的完整补偿输出并不因此固定为端点输出。

配置采用 `Selector.anchor_feature_dim: 0`、`Selector.anchor_ratio_features: 0`、`Selector.compensation_method: "s00r00e08d12"`。其他方法参数、Loss、Ratio轮转、Joint、TF、SI/GI、停用、pruning及增长规则不变。Mapping上限仍为0，补偿启用时刻仍为3000 Mapping步。

## Compensator容量研究已经覆盖什么

矩阵初筛覆盖：额外latent维度0/4/8/14/28，补偿私有隐藏层0/1/2，隐藏宽度固定32。初筛4000步、2seed；入围和单变量控制12000步、8seed。以下为同一fr1冻结地图、同K、留出Bank视图的配对差值：

| 单变量改变 | 固定条件 | 平均ΔPSNR | 95%配对区间 | 判断 |
|---|---|---:|---|---|
| 补偿头0→1个私有隐藏层 | S16/R0/E14，Selector私有头1层 | +0.552 dB | [+0.013, +1.041] | 非线性补偿头优于纯线性解码的方向；6/8seed改善 |
| 补偿头1→2层 | S16/R0/E14，Selector私有头1层 | +0.206 dB | [-0.241, +0.682] | 区间跨0，未证实稳定收益 |
| 额外latent8→14维 | S16/R0，两个私有头分别1/2层 | +0.551 dB | [+0.121, +0.950] | 7/8seed改善，说明更高维在此组合有效 |
| Selector latent0→16维 | R0/E8，两个私有头分别1/2层 | -0.905 dB | [-1.381, -0.445] | 同样的8维补偿输入在不同共享结构下结果不同 |

这些是Bank内探索性比较；7项比较的Holm校正后，不将上述区间直接解释为全部因子的确证性结论。完整原始结果为 `tests/diagnostics/selector_capacity_matrix_20261026/evidence/analysis/matched_confirm_controls.csv`。

S00R00E08D12是本轮复验组合的均值首名，不能据此断言“8维和2层已经到容量上限”。尚未正式完成固定S00R00、Selector头1层条件下，仅改变E8→14/28，或仅将补偿头2→3层的配对复验；初筛中不同结构的差值不能替代这种单变量证据。本轮不新增这些训练。

## 接入验证与边界

- C++ CPU原生核验通过：11/40/14形状、初始p=0.5与零补偿、补偿梯度、非活跃ratio无梯度和Adam推进。
- 原生新增、prune、闭环按成员数加权迁移、Adam状态扩展、teacher clone、保存/重载通过；四张8维补偿表随Anchor生命周期维护。
- C++/Python在1/2.5/5/7.5/10/12.5/15%上的概率、共享编码和选后补偿一致；latent插值和端点查询通过。
- 旧16/32维、ratio-feature和R3格式加载核验通过；新格式显式使用Anchor布局v10和补偿latent布局v2，不截断旧模型参数来伪装兼容。
- OpenCV实际解析默认和R3参考YAML通过。新11维输入使用LibTorch通用输入打包，原27维CUDA专用kernel不套用于该布局；14维属性CUDA解码接口保持原样。
- 本轮未进行完整CUDA Mapping构建或新的在线SLAM质量训练。此前矩阵是冻结地图实验，补偿从第1个LoD步启用；新在线默认继续原来的3000 Mapping步warmup，不能把矩阵收益当作在线收益。
- 未stage、commit或push。
