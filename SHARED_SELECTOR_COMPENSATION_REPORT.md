# 共享浅层 Selector＋选后补偿：实现与质量测试

**结论：结构已实现并通过功能验证；补偿头有明确作用，但本次替换 TF 后未改善整体重建质量。** 同一新模型、同一选中集合下，补偿显著改善图像；与原可变成员＋TF 基线相比，Full 和四档原 ratio 的四指标均值均退化。same-K 仅最低预算改善，其余三个预算退化。不能据此判定 Anchor Feature 无法替代六平面，也不能将功能通过表述为质量成功。

## 实现

输入保持 27 维：Anchor Feature 16＋局部 Offset 3＋activated Scale 3＋SH DC 3＋activated Opacity 1＋log(ratio) 1。共享编码为 `27→32 ReLU`；私有选择分支为 `32→32 ReLU→2 softmax`；私有补偿分支为 `32→32 ReLU→10`。

先对 N 个候选计算共享特征和选择概率，沿用 Cached Bernoulli 得到集合 S；随后只 gather `h[S]` 并解码 K 行补偿。缓存仅在当前 forward/backward 内存活。保持旧选择网络的参数命名与布局，旧 checkpoint 继续可读；新 checkpoint 含补偿 schema 和 head 权重。

补偿输出为 δXYZ、δlog-scale、δquaternion_raw；应用 `XYZ′=XYZ+δXYZ`、`scale′=scale·exp(δlog-scale)`、`q′=normalize(normalize([1,0,0,0]+δq)⊗q)`。只改变当前 LoD render 的属性，Master 继续显式存储。最后一层 weight/bias exact zero，初始化几何恒等，选择概率 exact 不变。

LoD 保持 joint：更新 Master Gaussian、共享层、两个私有分支和 Anchor Feature；Full 只更新 Master。补偿自 iteration 3000 开始，与基线 TF 启用时刻相同。共享参数由原 Selector optimizer ownership 管理，一次正常 backward 后更新一次。

本次关闭 TF，不创建 HexPlane 或 TF optimizer；原 HexPlane 正则随之移除。原 photo、ratio loss、Cached Bernoulli、ratio 调度、可变成员生命周期、停用功能和 SI/GI disabled 保持不变；没有新增 loss、feature、采样策略或正则。

几何 Change Distill 将原 TF 的补偿后几何目标运输并入 Selector 同一次 CD 更新，同时保持原选择概率目标；event-local Adam 不改正常 Adam 状态。当前实现要求 Selector/原 TF CD epochs 相等，本次均为 5。

## 固定实验协议

- `fr1_xyz / seed2026 / 10000 Mapping`，只训练一次候选；评价最终 checkpoint，无质量调参、无选最好 step。

- 基线复用 `/home/prmi/zwj/variable-anchor-joint-quality-20261021/variable_seed2026`，没有重训固定或串行 joint 模型。

- 原 joint：online 按 1%/5%/10%/15%/Full 轮转；tail 每轮 Full 后执行一个轮转 joint LoD。`serial_full_lod=0`，Full100=0。

- 使用此前锁定的 64 个共同 RGB 视图，各自在线估计位姿；不是新挑的最好视图，也不声称严格 non-Bank。

- 相同 K 从共同训练 ratio 区间内确定为 75/405/734/1064。每模型输入 ratio=K/N；以 cached logit margin 排 exact-K 仅作诊断，生产仍为 Cached Bernoulli。

- 补偿 OFF 诊断复用新模型权重、ratio 和逐行 exact 相同的选中集合，只在推理时关闭补偿。它不等于重新训练一个无补偿模型。

## Full-map 质量

| 方法 | Gaussian | PSNR↑ | SSIM↑ | LPIPS↓ | photo↓ |
|---|---:|---:|---:|---:|---:|
| 旧 Selector＋TF | 7099 | 21.7957 | 0.75245 | 0.29203 | 0.08713 |
| 共享 Selector＋补偿 | 7406 | 21.5113 | 0.74507 | 0.30666 | 0.08991 |

Full PSNR 下降 0.2845 dB，改善仅 15/64 视图；LPIPS 退化 0.01463，改善仅 5/64。Full 推理不执行选择或补偿，差异来自联合在线训练后的 Master 状态。

## 生产 Cached Bernoulli 质量

| ratio | 方法 | 实际 K | PSNR↑ | SSIM↑ | LPIPS↓ | photo↓ |
|---|---|---:|---:|---:|---:|---:|
| 1% | 旧 TF | 89 | 15.8053 | 0.58157 | 0.70884 | 0.16567 |
| 1% | 共享补偿 | 73 | 15.2089 | 0.56679 | 0.72361 | 0.17684 |
| 5% | 旧 TF | 350 | 18.2414 | 0.63271 | 0.58842 | 0.13247 |
| 5% | 共享补偿 | 364 | 17.9123 | 0.62163 | 0.60354 | 0.13831 |
| 10% | 旧 TF | 703 | 19.4241 | 0.66333 | 0.51015 | 0.11873 |
| 10% | 共享补偿 | 727 | 18.8811 | 0.64903 | 0.54145 | 0.12538 |
| 15% | 旧 TF | 1052 | 19.6795 | 0.67351 | 0.48005 | 0.11501 |
| 15% | 共享补偿 | 1065 | 19.0030 | 0.65751 | 0.52082 | 0.12255 |

四档 ΔPSNR 为 −0.5964 / −0.3290 / −0.5430 / −0.6764 dB；实际预算不完全相同，需结合下面的 same-K 表判断。

## 相同 K：新方法减旧 TF

| K | 旧 PSNR | 新 PSNR | ΔPSNR | ΔSSIM | ΔLPIPS | Δphoto | PSNR 改善视图 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 75 | 14.7022 | 15.1606 | +0.4584 | +0.00795 | -0.00075 | -0.00893 | 49/64 |
| 405 | 18.4069 | 18.0175 | -0.3893 | -0.01450 | +0.02855 | +0.00713 | 16/64 |
| 734 | 19.4540 | 18.8711 | -0.5829 | -0.01660 | +0.03586 | +0.00771 | 9/64 |
| 1064 | 19.6713 | 18.9923 | -0.6790 | -0.01639 | +0.04157 | +0.00767 | 8/64 |

K=75 四指标均值改善，但 LPIPS 中位数退化、仅 25/64 视图改善；其余三个 K 的四指标均值均退化。结论为 **same-K MIXED，主要预算退化**。

## 补偿是否真的有效：同模型、同一选中集合 ON 减 OFF

| ratio | K | OFF PSNR | ON PSNR | ΔPSNR | ΔSSIM | ΔLPIPS | Δphoto |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1% | 73 | 8.9880 | 15.2089 | +6.2209 | +0.20268 | -0.04416 | -0.15152 |
| 5% | 364 | 13.4975 | 17.9123 | +4.4148 | +0.14552 | -0.01263 | -0.07891 |
| 10% | 727 | 15.4135 | 18.8811 | +3.4675 | +0.13019 | -0.00741 | -0.06151 |
| 15% | 1065 | 15.9587 | 19.0030 | +3.0443 | +0.10976 | -0.00570 | -0.05183 |

四档的 PSNR、SSIM、photo 均为 64/64 视图改善，LPIPS 改善比例分别为 100%/76.6%/65.6%/62.5%。说明头部学习到了有效补偿，同时联合训练后的 LoD 已依赖该补偿；这不是相对旧 TF 的收益。

补偿幅度也较大：1% 时 δXYZ 的逐分量 RMS 为 0.3567、最大绝对分量 2.4077；15% 时为 0.0883、2.1012（世界坐标单位，基础 h=0.1）。1% 的 δlog-scale RMS 为 0.6677。这不是几乎零输出或完全没更新，但目前证据不足以定位是共享梯度、学习率、表示还是目标造成质量差距。

![质量曲线](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/shared_compensation_validation/quality_curves.png)

## 训练轨迹与初步成本

| 项目 | 旧 TF | 共享补偿 |
|---|---:|---:|
| Gaussian | 7099 | 7406 |
| Anchor | 1798 | 1775 |
| Full Render | 4743 | 4470 |
| LoD Render | 8686 | 8618 |
| Master Adam 更新 | 13429 | 13088 |
| Selector 更新 | 8686 | 8618 |
| TF 更新 | 6286 | 0 |
| online 结束 Mapping | 6571 | 6912 |
| Selector＋补偿参数，不含 Feature | 2018 | 3404 |
| Anchor Feature 标量数 | 28768 | 28400 |
| TF 参数 | 1186762 | 0 |

候选补偿执行 6218 次，累计解码 3,310,927 行；本次没有发生补偿 CD（0 步）。L0/L1/L2 Anchor 分布：旧 [627, 1157, 14]，新 [600, 1156, 19]。

10% 下 Python 完整选择＋补偿的五组均值：旧 4.358 ms，新 2.800 ms；临时峰值 allocated 增量分别为 5.479 / 5.184 MiB。包含统计/同步与各自 N/K，不是固定人口的原生 SLAM FPS 对照。TF 约 118.7 万参数被 1386 个新增 head 参数替代；此项参数节省是确定的。

候选训练 wall time 64.12 秒；基线已存记录 68.61 秒。异步 online 结束时间和实际更新次数不同，因此不据此声称训练加速或将质量差距单独归因于网络结构。

## 验证状态与边界

- **IMPLEMENTED / TESTED**：共享编码、私有选择头、选后 K 行补偿、TF 关闭、原 joint/预算/lifecycle、保存加载和导出接入。

- **TESTED**：CPU/CUDA 零初始化恒等、原选择概率 exact 一致、不污染 RNG；补偿梯度到达 head/trunk/Feature；真实 renderer photo 反传、Full 参数与 Adam 隔离；原 online/tail 调度；旧可变成员/Mapping/CD 回归。

- **TESTED**：最终真实 checkpoint 的四档 C++/Python mask exact 一致；补偿后 XYZ/Scale/Rotation 与 PLY 导出数值一致；Master SH/Opacity 沿用显式属性；模型和源文件 hash 冻结。

- **PARTIALLY VERIFIED**：补偿几何 CD 已通过合成局部几何修正、教师冻结和正常 Adam 不变测试；本次真实训练的补偿 CD 为 0，不能声称已验证真实 Loop Closure 质量。

- **NOT TESTED**：多场景、多 seed、真实 Loop Closure 压力测试、原生端到端性能；没有开展学习率或容量搜索。

- 补偿头使用现有 Selector Adam，LR≈0.001；原 TF MLP 使用独立且更低的 LR 时钟。此轮没有匹配两者 LR 的消融。架构、补偿参数化、优化尺度和移除 HexPlane 正则共同变化，当前结果不能证明某一个因素单独失败。

- 本地已复算 22 条评价、1408 条逐视图记录、52 条配对统计；核对 334 个证据文件、177 个源文件和 58 个基线复用文件的 SHA256。

## 交付

- [启用配置](H:/AIProjects/cartgs优化更细致版/mycartgs/cfg/selector/anchor_compensation.yaml)：共享补偿开启、TF 关闭；未强制覆盖历史配置。

- [质量汇总](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/shared_compensation_validation/evidence/validation/quality_summary.csv) / [逐视图](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/shared_compensation_validation/evidence/validation/quality_per_view.csv) / [配对统计](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/shared_compensation_validation/evidence/validation/paired_statistics.csv)

- [补偿幅度](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/shared_compensation_validation/evidence/validation/compensation_statistics.csv) / [训练与成本](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/shared_compensation_validation/evidence/validation/model_diagnostics.json) / [功能门控](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/shared_compensation_validation/evidence/validation/correctness.json) / [交付核验](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/shared_compensation_validation/local_delivery_checks.json)

- [最终 Selector＋补偿权重](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/shared_compensation_validation/evidence/candidate_seed2026/10000_shutdown/ply/selector.pt)

完成后停止，没有调参、扩网、新损失、恢复 TF 或追加训练；没有 stage、commit、push。
