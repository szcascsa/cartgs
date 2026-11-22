# 仅 Anchor Pruning 与满10点下一层增长：质量消融

**两组实验已完成，逐 Gaussian 质量 pruning 删除数均为 0。** 相同 K 下，A 相对旧基线有 2/4 档同时改善 PSNR 与 photo；B 相对 A 有 0/4 档同时改善。完整指标和人口差异列于下方，不能仅按原 ratio 均值归因。

**当前判断：保留 A 作为本轮请求恢复的模式；不建议启用“满10才向下一层增长”的 B 门槛。** A 的四档原 ratio 四指标均值均改善，但 same-K 为 MIXED，不能声称选择质量在所有预算提高。B 相对 A 的四档原 ratio 和四档相同 K 四指标均值均退化，门槛本次没有带来 LoD 收益。这只是单场景、单 seed 证据。

## 三组定义

| 方法 | Gaussian 逐点 pruning | Anchor 窗口 pruning | 下一层 growing | 同层 growing |
|---|---|---|---|---|
| 旧基线 | 原 opacity／尺寸规则 | 原规则 | 原梯度规则 | 原梯度规则 |
| A | 关闭 | 原规则 | 原梯度规则 | 原梯度规则 |
| B | 关闭 | 原规则 | 原梯度规则＋源 Anchor 恰好10成员 | 原梯度规则 |

按用户确认，B 仅限制下一层 growing，同层 growing 不要求满10。新观测的多尺度 Anchor 初始化继续执行；因此部分细层 Anchor 可以由新观测直接创建，不能仅凭细层 Anchor 存在就判定门槛失败。

保留 Gaussian clone/split 用于填充现有 Anchor，保留 Gaussian／Anchor 拓扑交替。split 仍以子点替换父点，这是密度更新事务，不是 opacity／尺寸质量 pruning；日志及计数将其分开。

Anchor 删除继续使用原窗口规则：可见访问累计所有成员的正 opacity 总和 O、访问数 T；仅 `T>check_interval·0.8` 且 `O<0.005·T` 才整 Anchor 删除。窗口重置、逐 slot 统计和停用规则不变。

## 实现与配置

- `Selector.anchor_only_pruning=1`：关闭 Gaussian 回合按 opacity／尺寸判定的逐点 pruning；clone/split 和原 Anchor 回合保留。

- `Selector.anchor_refine_requires_full=1`：只在生成下一层新 Anchor 候选时检查源 Anchor 成员数；同层候选、既有父点归属及新观测插入不变。

- 按用户要求，[默认实验配置](H:/AIProjects/cartgs优化更细致版/mycartgs/cfg/selector/selector_config.yaml) 与 [命名补偿配置](H:/AIProjects/cartgs优化更细致版/mycartgs/cfg/selector/anchor_compensation.yaml) 均设为 A；现有启动脚本默认读取前者。两份配置参数一致，方法参数沿用 A 实际训练配置：在线 LoD scale=0.5、ChangeDistill epochs=5、补偿 start3000、TF/SI/GI disabled、原 joint。默认 `Selector.max_mapping_iterations=0`，沿用原 CaRtGS 停止规则；下文消融实验的10000步仅是当时测试预算，不作为默认强制上限。TF 的保留参数也对齐实际配置，但 TF 仍关闭。B 只通过显式配置开启第二个开关；历史配置不写新开关时沿用旧行为，可继续回归。

- 保存配置和 Anchor 状态中记录两项策略、逐点与整 Anchor 删除数；growing 日志记录 full／partial 来源及被门槛拦截数。

## 锁定协议

后续默认配置更新：按用户要求关闭 ChangeDistill（`ChangeDistill.enable=0`，缺省值同样关闭）。下文质量实验仍对应当时冻结的配置，不据此次默认开关调整重新解释历史结果。

- `fr1_xyz / seed2026 / 10000 Mapping`，只新增训练 A、B 各一次；旧14维模型复用 `/home/prmi/zwj/shared-selector-appearance-20261021/candidate_seed2026`，不重训。

- 27维输入、共享浅层＋私有选择分支＋14维几何／Opacity／SH DC补偿、零初始化、start3000、Adam／LR、重建和预算损失不变。

- 原 joint，`serial_full_lod=0`、Full100=0，TF关闭、SI/GI disabled，maturity=1、Cached Bernoulli与停用功能保留。online五档轮转，tail为Full后一个轮转joint LoD。

- 评价最终10000 checkpoint，无调参或选最好step；沿用64个共同RGB帧，各自在线估计位姿，不宣称严格non-Bank。

- 三组共同整数K=[119, 457, 796, 1134]，由 `[ceil(.01·max N), floor(.15·min N)]` 均匀取四点。每组输入ratio=K/N，均在1%～15%范围；cached margin exact-K仅作诊断，不替换生产采样。

## 人口、拓扑和成本

| 项目 | 旧基线 | A | B |
|---|---:|---:|---:|
| Gaussian | 7563 | 11825 | 11172 |
| Anchor | 1863 | 3062 | 2786 |
| 满10成员 Anchor | 346 | 499 | 505 |
| 平均成员数 | 4.0596 | 3.8619 | 4.0101 |
| Master opacity <1/255 | 0 | 3934 | 3551 |
| Master opacity <0.02 | 0 | 4267 | 3890 |
| 窗口停用点 | 0 | 1937 | 1860 |
| 逐点质量 pruning 删除成员 | 旧模型未记录 | 0 | 0 |
| 整 Anchor pruning 删除成员 | 旧模型未记录 | 6014 | 6028 |
| clone 数 | 1 | 4 | 0 |
| split 父点数 | 6290 | 4836 | 4850 |
| 下一层候选 Gaussian 数 | 806 | 491 | 208 |
| 未满10拦截的下一层候选 | 0 | 0 | 242 |
| 实际放行的部分成员来源下一层候选 | 0 | 228 | 0 |
| 梯度 growing 新 Anchor | 1060 | 647 | 355 |
| 原生 Selector forward ms | 0.1683 | 0.1684 | 0.1736 |

新旧基线的 fine 来源细分计数日志格式不同，旧模型此项记为0表示未记录，不作为门槛证据；B 的实际放行部分成员来源候选必须为0。原生forward在各自实际N上测量，不能视作相同N的算法性能比。

## Full 与生产 Cached Bernoulli

| 档位 | 方法 | 实际K | PSNR↑ | SSIM↑ | LPIPS↓ | photo↓ |
|---|---|---:|---:|---:|---:|---:|
| Full | 旧：逐点＋Anchor pruning | 7563 | 21.5785 | 0.75144 | 0.29917 | 0.08815 |
| Full | A：仅 Anchor pruning | 11825 | 21.7515 | 0.75084 | 0.30275 | 0.08709 |
| Full | B：仅 Anchor pruning＋满10才向下增长 | 11172 | 21.8167 | 0.75482 | 0.30059 | 0.08576 |
| 1% | 旧：逐点＋Anchor pruning | 75 | 14.9774 | 0.57235 | 0.71847 | 0.17739 |
| 1% | A：仅 Anchor pruning | 120 | 16.4484 | 0.59005 | 0.70205 | 0.15895 |
| 1% | B：仅 Anchor pruning＋满10才向下增长 | 103 | 15.4701 | 0.57807 | 0.70797 | 0.16982 |
| 5% | 旧：逐点＋Anchor pruning | 362 | 17.8645 | 0.62868 | 0.61035 | 0.13644 |
| 5% | A：仅 Anchor pruning | 578 | 18.9112 | 0.65143 | 0.53967 | 0.12389 |
| 5% | B：仅 Anchor pruning＋满10才向下增长 | 542 | 18.3800 | 0.63523 | 0.58071 | 0.13264 |
| 10% | 旧：逐点＋Anchor pruning | 727 | 19.2360 | 0.66130 | 0.52326 | 0.12042 |
| 10% | A：仅 Anchor pruning | 1162 | 19.6878 | 0.67469 | 0.48048 | 0.11430 |
| 10% | B：仅 Anchor pruning＋满10才向下增长 | 1090 | 19.0078 | 0.65624 | 0.51532 | 0.12334 |
| 15% | 旧：逐点＋Anchor pruning | 1069 | 19.5425 | 0.67004 | 0.49490 | 0.11687 |
| 15% | A：仅 Anchor pruning | 1710 | 19.8220 | 0.68208 | 0.46138 | 0.11167 |
| 15% | B：仅 Anchor pruning＋满10才向下增长 | 1633 | 19.2550 | 0.66510 | 0.49573 | 0.11961 |

原ratio的K随地图人口变化，不能把更多点带来的收益直接归因于取消逐点pruning。

## 相同K：三组对照

| K | 比较（前减后） | ΔPSNR | ΔSSIM | ΔLPIPS | Δphoto | PSNR改善视图 |
|---:|---|---:|---:|---:|---:|---:|
| 119 | A−旧 | +1.5078 | +0.02751 | -0.00974 | -0.02366 | 64/64 |
| 119 | B−旧 | +0.1742 | +0.01072 | -0.00578 | -0.00730 | 43/64 |
| 119 | B−A | -1.3337 | -0.01678 | +0.00396 | +0.01636 | 1/64 |
| 457 | A−旧 | -0.2686 | +0.00123 | -0.01016 | +0.00040 | 26/64 |
| 457 | B−旧 | -0.8270 | -0.01742 | +0.02275 | +0.01108 | 3/64 |
| 457 | B−A | -0.5584 | -0.01865 | +0.03290 | +0.01068 | 9/64 |
| 796 | A−旧 | -0.0226 | -0.00163 | -0.00340 | -0.00001 | 33/64 |
| 796 | B−旧 | -0.7073 | -0.01793 | +0.02683 | +0.00895 | 8/64 |
| 796 | B−A | -0.6848 | -0.01631 | +0.03023 | +0.00896 | 10/64 |
| 1134 | A−旧 | +0.1593 | +0.00358 | -0.00711 | -0.00250 | 41/64 |
| 1134 | B−旧 | -0.4808 | -0.01279 | +0.02182 | +0.00580 | 13/64 |
| 1134 | B−A | -0.6400 | -0.01638 | +0.02893 | +0.00830 | 10/64 |

A−旧的四档 same-K ΔPSNR 为 +1.5078 / −0.2686 / −0.0226 / +0.1593 dB。B−A 为 −1.3337 / −0.5584 / −0.6848 / −0.6400 dB，PSNR 改善视图仅 1/9/10/10 个（每档64视图）；不能把 B 的 Full 小幅改善解释为 LoD 改善。

## 低 Opacity 成员与预算使用

关闭逐点 pruning 后，低 Opacity 存储点重新出现：A 有3934个、B有3551个 Master opacity低于1/255。停用分别覆盖1937／1860个点；其余点可能尚未满足完整可见窗口资格，不能将未观察到的访问算作零opacity。

| ratio | 方法 | 实际K | 选中Master opacity<1/255 | 补偿后opacity<1/255 |
|---|---|---:|---:|---:|
| 1% | A | 120 | 0 | 0 |
| 5% | A | 578 | 1 | 1 |
| 10% | A | 1162 | 23 | 23 |
| 15% | A | 1710 | 119 | 112 |
| 1% | B | 103 | 1 | 1 |
| 5% | B | 542 | 3 | 3 |
| 10% | B | 1090 | 44 | 37 |
| 15% | B | 1633 | 172 | 151 |

15%时补偿后仍低于截断的选中点为 A 112/1710（6.55%）、B 151/1633（9.25%）。这是该checkpoint的opacity诊断，未新增候选屏蔽或修改停用阈值，也不把它等同于逐点真实图像贡献测量。

![质量曲线](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/anchor_pruning_ablation/quality_curves.png)

## 在线训练轨迹与验证边界

| 项目 | 旧基线 | A | B |
|---|---:|---:|---:|
| Full Render | 4612 | 4522 | 4584 |
| LoD Render | 8654 | 8631 | 8646 |
| Master Adam更新 | 13266 | 13153 | 13230 |
| Selector更新 | 8654 | 8631 | 8646 |
| online结束Mapping | 6734 | 6847 | 6770 |
| 补偿前向 | 6254 | 6231 | 6246 |
| 补偿CD步数 | 0 | 0 | 0 |
| Anchor Feature标量数 | 29808 | 48992 | 44576 |
| 显式Gaussian参数bytes | 1784868 | 2790700 | 2636592 |

A/B训练wall time分别为 65.24 / 66.06 秒；不是隔离性能基准。显式参数bytes不含Adam、梯度、SLAM或renderer临时显存，未测完整训练峰值显存。

- **IMPLEMENTED / TESTED**：两项独立开关、保存配置、原Anchor prune规则、逐点prune关闭、clone/split保留、下一层门槛与同层不受限、原joint/Full梯度隔离。

- **TESTED**：CPU/CUDA策略单测及Mapping接入，14维补偿与旧10维checkpoint兼容回归，共17项门控通过；最终真实模型C++/Python/Cached Bernoulli mask exact一致，补偿属性与PLY导出数值一致，source与模型hash冻结。

- **PARTIALLY VERIFIED**：真实训练的门槛日志及无逐点prune计数通过；这里只验证一个scene/seed，未做跨场景和多seed稳定性实验。异步SLAM导致在线结束时刻、地图人口、位姿与实际优化次数不同，不能作单因素固定地图因果归因。

- 本地重算27条评价、1728条逐视图和108条配对统计；422个证据文件、179个源文件及58个旧评价文件hash核验通过。

- 默认配置切换后另行核验：OpenCV 成功解析两份配置，112个字段中111个在生产 FP32 精度下与 A 的实际输入配置一致；唯一有意差异是默认 Mapping 上限从测试的10000恢复为0，遵守用户要求。原179个源码文件中177个哈希不变，仅两份默认配置按本次要求修改。CPU checkpoint／选后解码／PLY 导出回归及33个 Python 文件语法检查通过，`git diff --check` 通过。本次未重新训练或重跑 GPU 门控；原生17项门控沿用上述冻结运行源码的证据。归档实验脚本中的原始源码 SHA 仍对应当时快照，不将默认配置的后续改动伪写进历史实验。

## 交付

- [质量汇总](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/anchor_pruning_ablation/evidence/validation/quality_summary.csv) / [逐视图](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/anchor_pruning_ablation/evidence/validation/quality_per_view.csv) / [配对统计](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/anchor_pruning_ablation/evidence/validation/paired_statistics.csv)

- [人口与拓扑](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/anchor_pruning_ablation/evidence/validation/model_diagnostics.json) / [增长事件](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/anchor_pruning_ablation/evidence/validation/growth_events.csv) / [选中点opacity](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/anchor_pruning_ablation/evidence/validation/selected_opacity.csv)

- [功能门控](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/anchor_pruning_ablation/evidence/validation/correctness.json) / [交付核验](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/anchor_pruning_ablation/local_delivery_checks.json)

- [A模型](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/anchor_pruning_ablation/evidence/anchor_only_seed2026/10000_shutdown/ply/selector.pt) / [B模型](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/anchor_pruning_ablation/evidence/anchor_only_full10_seed2026/10000_shutdown/ply/selector.pt)

质量消融阶段没有额外调参、改loss或追加实验，也未提交代码。随后按用户要求将 A 设置为实验默认，并将相关实现、配置、测试及报告纳入本地代码提交；模型、渲染和其他生成产物不纳入提交，不 push。
