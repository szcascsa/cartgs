# Full＋全局 LoD1＋局部 LoD2：实现与验收报告

日期：2026-11-05，Asia/Hong_Kong。

## 结论与范围

已按确认设计实现三分支联合训练，独立编译完成，11项原生验收全部通过，并在真实 fr1_xyz RGB-D Full checkpoint 上完成基线、联合主方案、额外全局分支对照各12步运行。Selector、Compensator、Feature 均共享；局部采样使用原 Gumbel-ST 和预算损失，无训练 Top-K、局部 GI 或新增网络参数。每步仍只提交一次共享 Adam。

本报告交付实现与训练语义验收，不是完整质量实验。12步不足以评价1%/5%压缩质量，尚无“局部 PSNR 提升且全局质量保持”的结论，也未将该研究方案切换为生产默认。

## 源码与服务器隔离

- 当前源码快照：`3a6c0c3f9aa670985e3fc80a2a28f76e690e41d6`，已在前序步骤选择性提交。
- 研究分支：`research/global-view-joint-lod-20261105`。
- 独立本地 worktree：`H:\AIProjects\cartgs优化更细致版\mycartgs-global-view-joint-lod-20261105`。
- 服务器研究目录：`/home/prmi/zwj/experiments/global_view_joint_lod_20261105`。
- 新实现保留为可审查的分支修改，未再次自动 commit、未 push。
- 原 `mycartgs` worktree 的四个既有矩阵研究修改保持未提交，未回退或覆盖。

服务器 `source/`、`objects/`、`runtime/bin/`、`runtime/lib/`、`configs/`、`runs/`、`logs/`、`reports/` 分开保存。所有 Mapper C++ 编译单元使用本研究快照重编译，避免新旧类布局混用。项目 LibTorch CUDA11.8 ABI1、g++9；复用原已验收且本轮未修改的 CUDA 静态库、Rasterizer 与第三方预编译库。第三方头文件只读复用，不修改共享 conda。

`ldd.txt` 及三组 `/proc/<pid>/maps` 均确认实际加载本研究 `runtime/lib/libgaussian_mapper.so`；运行时优先加载本研究动态库目录。未向主项目 bin、lib 或 results 输出文件。

## 训练目标与候选范围

每步选择同一训练相机、同一轮转 ratio，在同一更新前状态下执行：

1. Full：未补偿 Master 的原重建目标，保留原 Full Opacity 正则。
2. LoD1：全图候选，保留原 Photo＋Budget＋全训练视角 FlexGS GI。
3. LoD2：名义 Anchor AABB 与当前视锥相交的全部成员，独立 Gumbel-ST 采样，Photo＋Budget；不从 LoD1 集合继续筛选。
4. 检查全部梯度的有限性，提交一次共享 Adam。

总损失为 `L_full + L_lod1 + L_lod2`，三分支外层权重1:1:1；Photo 系数保持1，Budget 系数保持0.01，全局 GI 保持1，局部 GI 为0。Photo 沿用原 L1／Smooth-L1 设置和 DSSIM 权重，不新增局部 Opacity 正则。相对原 Full＋LoD，增加了 LoD 总监督权重与计算，需在质量实验中单独归因。

两路预算函数都为：

`L_budget = abs(sum(mask) - ratio * candidate_N) / max(1, ratio * candidate_N)`。

局部 candidate_N 包含可见 Anchor 的 inactive 成员，Slot 资格仅屏蔽实际选择；目标保留浮点值，不取整、不裁剪到 eligible 数。记录候选数、eligible 数、浮点目标、实际 K、预算不可达部分及实际误差。原配置的预算归一化开关仍可回退，两路使用同一个开关。

名义视锥不等于真实像素贡献判定。其中心／尺度沿用现有 Viewer 的 Anchor 名义格子；边界相交保留，Mask 与候选行索引 detach。当前 Master 或补偿后的 Gaussian 可移出名义格子，相关裁剪损失仍是独立限制。

## 网络、梯度和随机数

保留原 Selector 11→32→32→2，以及 Offset3＋累加 Residual48＋连续 Ratio3、64/64、四个 Head 的 Compensator。未添加相机编码、第二套 Feature、保留率条件或空间稀疏增强。Scale 继续使用原 checkpoint/config 的 Softplus 解码。

局部函数重新读取更新前 Master 并独立构图，保留原 Master 物理输入和 Feature Gather/Scatter 梯度。它不调用 zero_grad 或 optimizer step；Full 只影响 Master，全局与局部 LoD 梯度在原共享优化器上累积。原全局函数内部的 Selector 清梯度仍位于局部分支之前，不会清除局部或已累积的 Full Master 梯度。

局部使用独立推进的 Generator，默认 seed0；每步重新采样，不固定整段训练噪声，不消耗原全局 CPU/CUDA RNG。保存 `view_budget_rng.pt` 供审计／恢复测试；并未宣称新增完整 SLAM、Adam、相机游标恢复功能。

无局部候选时跳过该分支渲染并清除连续零选点／饱和检查历史；Full 与全局仍更新。有候选但抽到零点时保留原空集合渲染与 Budget 反传，不重采样凑 K。局部健康检查独立计数，连续零选点、概率饱和、异常 Scale／XYZ 长尾在 Adam 前终止，失败文件明确记录 `branch=view_budget`。

## 改动文件与启用方式

| 文件 | 改动 |
|---|---|
| `src/view_budget_training.cpp` | 局部候选、私有采样、补偿、Photo／Budget 反传及诊断 |
| `include/view_budget_training.h` | 复用原 CPU/CUDA Gumbel-ST 采样器 |
| `src/real_lod.cpp` | 第三分支接入、一次 Adam、总损失与真实曝光记录 |
| `src/gaussian_mapper.cpp`、`include/gaussian_mapper.h` | 配置、兼容限制、私有 RNG 与计数状态 |
| `cfg/selector/spatial_offset3_residual48_global_view_joint.yaml` | 独立主实验模板，默认四档0.2/0.4/0.6/0.8，继承生产模板；真实 checkpoint 验收显式使用六档 |
| `tests/anchor_selector/view_budget_training_test.cpp`、`CMakeLists.txt` | CPU/CUDA 和 Mapping 验收目标 |
| 本目录研究脚本 | 独立打包、编译、验收、真实 checkpoint 短程运行与逐位核查 |

主要开关为：

```yaml
ViewBudget.enabled: 1
ViewBudget.loss_weight: 1.0
ViewBudget.seed: 0
ViewBudget.global_control: 0
```

旧配置缺省关闭。`loss_weight=0` 时不执行局部分支且不生成局部随机数。`global_control=1` 仅把第三分支候选改为全图，作为增加一次 Photo＋Budget 的对照，不重复加入 GI。

启用模式要求原11D／Residual48、Joint Full＋LoD、全局无视锥筛选；与旧 ViewInput／ViewResidual／MixedRatio、Maturity／Gate 等实验组合显式互斥，避免无意改变既定实验定义。生产 `selector_config.yaml` 未修改。

## 原生验收

最终11项均返回0：

| 验收 | 检查内容与结果 |
|---|---|
| sampling | CPU/CUDA ST 前向、VJP、未取整预算、Inactive Slot、空输入、私有 RNG 推进和恢复通过 |
| joint | 8次 Full、8次全局、8次局部反传、8次 Adam；拓扑与 checkpoint 通过 |
| empty | 空视角不执行局部渲染，全局继续；已有健康历史正确重置 |
| control | 第三分支使用全图，预算与记录对应全图 N |
| off、off_repeat、zero | 关闭／零权重均不初始化局部 RNG、不执行第三分支；重复运行用于量化原路径数值波动 |
| gradient | 局部函数保留预置 Master／Selector／Compensator／Feature 梯度，不提交 Adam；本次最大累积误差0 |
| health | 人为连续零选点在 Adam 前停止，参数未变、诊断分支正确 |
| legacy_real_lod | 旧 Full 隔离、全视角 GI、Full＋LoD 累积、Joint Master 和 checkpoint 回归通过 |
| legacy_lod_health | 原健康检查回归通过 |

四张 Residual 表的 Adam step 为8／6／4／2，证明分支数量未改变非活动表跳过更新及每步只更新一次的语义。网络参数量无增加，保存再读的参数值相同。

关闭与零权重的8步终态存在23个 Tensor 位差，最大绝对差 `2.2351741790771484e-8`；相同关闭路径的独立重复运行也有23个 Tensor 位差，最大 `2.9802322387695312e-8`。CPU/CUDA 全局 RNG 终态均逐位相同。该结果支持原 CUDA 执行非确定性解释，但本轮未逐算子定位首个不同值，不能称为已证明差异来自某一个 Atomic Kernel，也不能宣称整个多步训练逐位可复现。

## 真实 Full checkpoint 验收

起点：`realLOD/saved_experiments/tum_full_tail_prune_20261103/jobs/freiburg1_xyz/rgbd/prune1/seed0_repeat2/attempt_01/model/11980_shutdown/ply`。Gaussian=13,298，Anchor=2,790，保存训练关键帧72。原 online=11,580、tail=400；本次仅把 Fine 验收目标设为12步，**未完成11,580步正式 Fine**。

三组均为 fresh Adam／RNG，从未训练 LoD 的 Full 起点初始化；六档训练条件为0.01/0.05/0.1/0.3/0.5/0.7，每档两步。相机半径重建 XYZ 学习率尺度为0.3976763785。这是固定拓扑 Full 起点继续优化，不是重新运行在线 SLAM，也不是完整优化器恢复。

| 组 | Full渲染／反传 | 全局LoD | 第三分支 | Adam | 第三分支候选数范围 | 第三分支实际K范围 |
|---|---:|---:|---:|---:|---:|---:|
| 基线 | 12 | 12 | 0 | 12 | — | — |
| 联合主方案 | 12 | 12 | 12 | 12 | 5,493–9,587 | 2,008–3,550 |
| 额外全局对照 | 12 | 12 | 12 | 12 | 13,298 | 4,969–5,153 |

三组 step0 Master SHA256 均为 `26e605baec3195e23088adc0fe29a52e0d1efd9d072762839c6b23299ea4ead5`；Selector／Feature／Buffer 的61个状态 Tensor 逐位一致；相机序列一致。新增分支没有新增可训练状态，六张 Feature 表沿用原初始化与累加规则。三组无非有限损失或训练失败，实际运行进程均加载独立研究 Mapper。训练图片输出数为0。

局部实际 K 远未贴近所有低 ratio 目标属于未训练起点的短程状态，不能据此判定最终预算收敛，也不隐藏该现象。后续必须观察完整 Fine 的实际K／目标K曲线、低ratio质量与健康检查。

12步平均 Mapping 时间记录为：基线33.19045 ms、联合36.63382 ms、额外全局36.00278 ms。这个短程包含首次重要性刷新、CUDA 预热且 GPU 存在无关共享进程，**仅作运行记录，不作为稳定开销或加速结论**。正式性能比较需充分预热、重复配对并核对实际 K。

## 主项目完整性与失败现场

对 `/home/prmi/zwj/mycartgs`、`/home/prmi/zwj/realLOD` 的关键源码、配置、bin 和 lib 共604个文件完成研究前后 SHA256 比较，604个全部一致，变化0。清单为 `PRODUCTION_BEFORE.json`、`PRODUCTION_AFTER.json`，核验摘要为 `PRODUCTION_INTEGRITY.json`。服务器剩余约40.5 GiB；未删除模型、未清理历史实验、未终止无关 GPU 进程。

首次构建发现研究源码目录缺少只读 simple-knn 头文件入口，补上只读链接；随后补齐本轮新文件的属性解码声明头。失败编译日志留存。短程核验脚本曾因零维状态 Tensor 的 Byte View 限制失败，改为展平后逐字节比较；未改变模型或放宽初始状态比较标准。

## 下一步

正式质量验证采用同一 Full 起点、相同相机序列、相同全局采样随机流与 Adam 步数，比较基线、额外全局对照、联合主方案。分别输出全局导出和局部严格预算推理的 PSNR／SSIM／LPIPS、实际K、低ratio分项与逐帧变化；单独检查 Full 质量及局部预算收敛，不能用训练总 Loss 代替评价。

质量筛查门槛在 `DESIGN.md` 中预先固定：全局训练 ratio 的平均 PSNR 不低于配对基线0.10 dB，任何训练 ratio 不下降超过0.20 dB，并单列1%／5%。增加局部损失会改变 Master 和 Compensator，三路目标冲突与额外监督强度均需通过对照分析。完整质量实验前保留生产默认。
