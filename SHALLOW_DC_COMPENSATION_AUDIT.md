# 浅层Encoder同时加回SH DC的质量对照

## 结论

- 加回DC相对不含DC浅层：两seed方向不一致，七档non-Bank same-K平均+0.054 dB（+0.206/-0.098）。
- 相对原Baseline：两seed均下降，平均-0.433 dB（-0.397/-0.468）。
- 相对原Baseline，训练ratio平均-0.388 dB、未见ratio平均-0.492 dB。

这是两个seed、同一冻结历史地图上的探索性结果，不能仅凭均值决定跨场景最优结构或声称在线Joint提升。原生产默认保留，没有根据本轮结果继续调整网络或训练。

**本轮没有恢复或超过原Baseline。** 若加回DC的两个seed方向不一致，不能把很小的平均正差当成稳定收益；原Cached实际预算结果与same-K结果同时报告。本轮不据此替换生产默认。


本轮在11→32 ReLU浅Encoder方案上，仅向Encoder和四Head直连输入增加SH DC3，网络实际变为14→32 ReLU。没有保持11维后挤掉其他输入；所有此前输入均保留。

## non-Bank same-K结果

| 参照 | ratio组 | seed3100 ΔPSNR | seed3101 ΔPSNR | 平均ΔPSNR |
| --- | --- | --- | --- | --- |
| original_baseline | all | -0.397 | -0.468 | -0.433 |
| original_baseline | train | -0.385 | -0.390 | -0.388 |
| original_baseline | unseen | -0.414 | -0.571 | -0.492 |
| shallow_no_dc | all | +0.206 | -0.098 | +0.054 |
| shallow_no_dc | train | +0.208 | -0.175 | +0.016 |
| shallow_no_dc | unseen | +0.203 | +0.004 | +0.103 |

| 方案 | PSNR↑ | SSIM↑ | LPIPS↓ | photo↓ |
| --- | --- | --- | --- | --- |
| original_baseline | 17.24435 | 0.60185 | 0.60022 | 0.15107 |
| shallow_no_dc | 16.75815 | 0.58916 | 0.60361 | 0.15902 |
| shallow_dc | 16.81184 | 0.58831 | 0.60042 | 0.15742 |

## 各ratio增量

| ratio | K | 相对不含DC浅层 | 相对原Baseline |
| --- | --- | --- | --- |
| 1% | 68 | +0.505 | -0.152 |
| 2.5% | 170 | +0.294 | -0.404 |
| 5% | 341 | -0.201 | -0.658 |
| 7.5% | 512 | +0.044 | -0.468 |
| 10% | 683 | -0.074 | -0.224 |
| 12.5% | 854 | -0.027 | -0.606 |
| 15% | 1025 | -0.165 | -0.515 |

## Bank、留出及原Cached评价

| 参照 | Bank训练视图ΔPSNR | 内部留出ΔPSNR | non-Bank实际Cached ΔPSNR |
| --- | --- | --- | --- |
| original_baseline | -0.444 | -0.454 | -0.556 |
| shallow_no_dc | +0.013 | +0.008 | -0.021 |

## 结构与初始化

- 输入顺序为[全局XYZ3,activated Scale3,sigmoid Opacity1,局部Offset3,logratio1,SH DC3]共14维；浅Encoder14→32 ReLU，输出32维共享编码。DC为原始SH系数，不转RGB、不额外归一化。
- 补偿输入raw14+每Anchor每ratio latent16+h32共62维，四Head仍各62→32 ReLU→32 ReLU→3/3/4/4。只对选中K行补偿，Selector不读取补偿latent。原14维输出含义与变换、ratio latent插值保留。
- Encoder前11列和四Head公共输入列/偏置/后续层与不含DC方案同seed复制，新增DC列全0；step0概率和零补偿exact一致，隐藏函数浮点容忍检查通过。
- 输入DC与原Baseline场景派生属性中的DC tensor exact一致；仅开放允许属性的代理和NaN旧派生输入检查通过，不依赖Rotation或其他隐式输入。
- 从真实零补偿起点的3次photo反传验证Encoder及全部四Head的新增DC连接随后获得梯度；其他ratio表grad=None。实际首步隐藏梯度为0属于零输出初始化，不误报未训练。
- Encoder参数384→480，四Head参数12366→12750，总参数146096→146576，新增480；因此本轮不仅增加信息，也增加输入连接，不能完全排除参数化作用。

## 协议与范围

- 同源fr1_xyz冻结N6838/M2066地图，Master/拓扑/Anchor归属/位姿/eligible/cached uniform不变。54 Bank训练320×240，54 Bank训练/13留出/45历史non-Bank评价640×480。
- 两seed各12000步、每ratio3000曝光，复用不含DC浅层和原Baseline各两个锁定最终模型；同帧序、ratio顺序、fresh fused Adam lr=.001 eps1e-15、原photo+.01预算Loss。
- 训练1/5/10/15%，另评2.5/7.5/12.5%，未见ratio不额外拟合。所有评价用最终12000步，未按non-Bank调整LR、步数或追加架构；4k/8k/12k曲线保留。
- primary same-K按logit(p)-logit(cached uniform)排名，K=floor(rN)并受eligible上限；实际Cached数量另存。Selector与Encoder联合训练，选择身份可变，不是固定选择的纯补偿因果。
- 原Baseline为Offset3+Scale3+SH DC3+Opacity1+logratio1，不含全局XYZ；候选还额外包含XYZ，且DC输入初始化不同，不能将与原Baseline的差值全归因于加回DC。
- 各参数Adam时钟按实际非空选择精确核验，不为公平伪造零梯度step；若存在空选择，辅助更新次数在fairness_checks中完整报告。
- 两seed是同一历史地图上的初始化/帧序重复；评价轨迹已多次用于研究，不能当全新盲测或声称跨场景、在线Joint收益。GPU有其他任务，耗时只记录不作性能结论。
- 生产源码/默认未替换，未stage、commit或push。本轮不做其他输入、深度、维度或LR搜索。

![质量曲线](tests/diagnostics/s00_shallow_dc_20261026/evidence/analysis/quality_curves.png)

## 内部Bank留出收敛

| 方案 | seed | 4000步 | 8000步 | 12000步 |
| --- | --- | --- | --- | --- |
| original_baseline | 3100 | 16.746 | 17.351 | 17.496 |
| original_baseline | 3101 | 16.721 | 16.887 | 17.307 |
| shallow_no_dc | 3100 | 16.702 | 17.036 | 16.987 |
| shallow_no_dc | 3101 | 16.329 | 16.604 | 16.892 |
| shallow_dc | 3100 | 16.125 | 16.834 | 17.123 |
| shallow_dc | 3101 | 16.467 | 17.140 | 16.773 |

全部使用固定最终12000步，有限预算差异不当作表示能力上限。

## 交付核验

2个新增模型、4个锁定参照均完整保存，全部12000步、每ratio3000次曝光。实际Adam时钟与非空集合记录精确匹配；DC输入、step0函数及真实photo梯度通过检查，源码与服务器归档SHA一致，逐视图四指标finite。Git HEAD、全部tracked diff和staged diff与实验前完全相同，已有17个tracked改动保留；未stage、commit或push。

新增模型空选择与补偿更新记录：[{"seed": 3100, "empty_steps": {"0.01": 0, "0.05": 0, "0.1": 0, "0.15": 0}, "compensation_updates": 12000}, {"seed": 3101, "empty_steps": {"0.01": 0, "0.05": 0, "0.1": 0, "0.15": 0}, "compensation_updates": 12000}]

[逐视图质量](tests/diagnostics/s00_shallow_dc_20261026/evidence/analysis/quality_per_view.csv) · [配对统计](tests/diagnostics/s00_shallow_dc_20261026/evidence/analysis/paired_statistics.csv) · [预算与时钟](tests/diagnostics/s00_shallow_dc_20261026/evidence/analysis/fairness_checks.csv)
