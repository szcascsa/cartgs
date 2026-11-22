# Full＋全局 LoD＋局部 LoD 联合训练设计

本研究在独立分支 `research/global-view-joint-lod-20261105` 实现。起点为源码快照 `3a6c0c3f9aa670985e3fc80a2a28f76e690e41d6`，不切换生产默认、不 push。此前草稿的局部 Top-K、固定噪声和局部 GI 已取消。

## 目标和主配置

每个 Fine step 使用同一相机、原轮转 ratio 和同一更新前参数，分别计算 Full、全局 LoD1、局部 LoD2。三路反传后提交一次共享 Adam。全局训练次数及原学习率时钟保持，增加一次局部渲染与反传。

主实验总损失为 `L_full + L_global + L_view`，权重 1:1:1。全局分支沿用原 Photo、预算和全训练视角 FlexGS GI。局部仅 Photo＋0.01×预算，不新增 Teacher、网络、可训练表、保留率条件或稀疏增强。Photo 及 Full Opacity 正则保持原配置，Full 正则不在局部分支重复计算。

Selector 仍为 physical11→32→32→2；Compensator 仍为 Offset3＋Residual48＋Ratio3，64/64 与四个 Head。Scale 解码沿用 checkpoint/config，保持所有 Master 物理输入与 Anchor Feature 梯度。

## 候选与预算

全局 LoD1 继续全图候选。局部先执行现有名义 Anchor AABB 六平面测试，取这些 Anchor 的全部 Master 成员，按原行序组成候选；不从 LoD1 已选点中再次抽样。边界相交也保留，Mask 和索引不反传。

令候选点数为 N，可选 Slot 数为 E，硬前向 Gumbel-ST Mask 为 m。两路都使用原采样器，不做 Top-K 或固定数量投影。预算损失为 `abs(sum(m)-r*N)/max(1,r*N)`，目标不取整，N 包含候选中的 inactive Slot，采样后屏蔽 inactive 点。E 小于 rN 时只记录预算不可达，不擅自改变目标。

局部逐步记录 N、E、浮点目标 rN、实际 K、预算偏差、Photo、Budget、加权损失、ratio、相机 ID。无候选时局部损失为零、跳过局部渲染，但 Full／全局训练仍执行。有候选但抽到零点时继续原零点渲染／预算反传，并由独立健康检测判断连续零选点，禁止重采样凑 K。

## 随机性与梯度

局部使用独立 Generator，默认 seed0，每步推进；不消耗全局 CUDA/CPU RNG。它不是 Viewer 的固定推理噪声。局部状态保存在 `view_budget_rng.pt`，用于审计和测试恢复；原完整 SLAM/Adam/RNG 恢复能力不因此改变。

Full、LoD1、LoD2 独立构图，顺序反传以释放渲染缓存。局部函数既不 zero_grad，也不更新参数。Selector、Compensator 和 Feature 的一次优化器时钟不因三个 backward 变成三次。非活动 Residual 表保留原 `grad undefined` 与 Adam 跳过语义。局部健康状态与全局分开计数，错误必须在共享 Adam 前终止并保存正确分支诊断。

同一点同 ratio 的确定性评分仍相同；局部任务改变候选集合和梯度监督，不构成相机条件预测。相同 ratio 的全局和局部选择集合可能要求不同补偿，联合监督不能保证同时改善。名义 Anchor 裁剪可能漏掉移出格子的有效点，此问题独立于联合训练。

## 开关与对照

旧配置不含 `ViewBudget.enabled` 时关闭，`loss_weight=0` 也不生成局部随机数、不执行第三分支。独立模板 `cfg/selector/spatial_offset3_residual48_global_view_joint.yaml` 开启，保留生产模板四档 ratio；真实场景六档测试从相同 Full 起点显式生成六档配置。

`ViewBudget.global_control=1` 仅作为额外全局分支计算量对照，第三分支仍是 Photo＋Budget，不另加 GI。正式评价需分别比较旧 Full＋全局、Full＋全局＋额外全局、Full＋全局＋局部，初始化、相机序列、Adam 次数、ratio 次序相同。局部分支增加训练计算及 LoD 总权重，不能把全部收益归因于视角候选。

## 完成标准和结果边界

1. 原生 CPU/CUDA 测试覆盖 ST 梯度、浮点预算、Slot、空候选、私有 RNG 推进与恢复。
2. 小规模真实 Mapping 检查三路累积、单次 Adam、Residual 表时钟、checkpoint 与拓扑、关闭开关及局部失败前终止。
3. 独立 LibTorch ABI1 构建通过，动态加载研究目录的 Mapper；生产关键文件前后哈希一致。
4. fr1_xyz RGB-D 未训练 Full checkpoint 短程运行通过，初始 Master/Selector 与对照逐位匹配、相机序列一致、零图片。

上述验收仅证明实现与训练语义。完整 Fine／全轨迹质量评价完成前，不宣称局部 PSNR 提升或全局质量保持。后续正式质量筛查预设：全局训练 ratio 平均 PSNR 不低于配对基线 0.10 dB，任一训练 ratio 不下降超过 0.20 dB，同时单独检查 1%/5%，不根据结果事后改阈值；单 seed 仍不足以作稳定性结论。
