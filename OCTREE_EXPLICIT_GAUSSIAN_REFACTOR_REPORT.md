# 多层 Anchor 固定十点显式 Gaussian 重构报告

## 范围与状态

基线为 `03e135b9083e49bd868ac776a9fa221504e2bb7d`。本轮新增独立的 `Selector.explicit_anchor_blocks: 1` 模式，配置见 `cfg/selector/octree_explicit.yaml`。旧固定/occupancy Anchor 路径保留作 baseline；新模式不使用旧 occupancy>128 细化。

用户已明确批准三项适配：只有十点全部满足原删除条件时才删整 Anchor；不足十个观测时立即有界初始化十点；几何修正后使用稳定 UID，允许一个 `(level,key)` 空间桶包含多个独立 block。它们不是未报告的算法变更。

**完成状态：IMPLEMENTED / TESTED。9 项正式功能 gate 全部通过；4 个新模型完成训练，36 项质量评价完成，共 792 个共同视图、28,512 行逐视图结果。29 个受控源码/协议文件 SHA 一致，评价前后模型 SHA 未变。单场景、两 seed 的 LoD 收益一致，但 Full-map 的 SSIM/LPIPS 存在回退，尚不能认定生产质量回归通过。**

## 地图与 Selector 的职责

地图初始化和在线增容现在先创建 Anchor，再一次性创建十个显式 Gaussian。`GaussianModel` 持有与 Selector 共享的 `AnchorBank`；Selector 只消费共享 Feature 和参考坐标，不能在 append/reset 回调中再次自行给点分配空间归属。

- 固定世界原点 `(0,0,0)`，FP64 floor 量化，`h0=0.1`。
- 最大 L2，三层尺度为 `0.1/0.05/0.025`。初始及新观测多尺度创建 L0/L1；L2 留给 Full 梯度增长。
- 每个有效 Anchor 恰有十个连续 slot；完整地图 `N=10M`。空间 key 负责查询和去重，稳定 UID 负责永久身份。粗细层可共存，父层点不迁移到新增细层。
- 六组 Gaussian 属性继续以扁平 Tensor 显式存储并独立优化。LoD compact input 只取行，不修改归属。
- 同层已占位置只跳过重复初始化；关键帧、RGB/深度图和现有训练仍正常接入。新模式取消旧“新增点一批至少十个”的过滤，单个新观测也能创建合法 Anchor。

27 维输入保持为 `Feature16 + localOffset3 + activatedScale3 + SH_DC3 + activatedOpacity1 + logRatio1`，网络仍为 `27→32 ReLU→32 ReLU→2`。局部 Offset 为 `(explicitXYZ-referenceCenter)/referenceScale`。本轮不输入 Global XYZ、camera、level、occupancy 或 SI/GI；显式 Gaussian XYZ 是唯一可优化几何。

新 checkpoint 布局为 v5，保存每节点参考尺度，因此不再把“cell-local offset”与 30 维 Global XYZ 开关绑在一起。旧 v2/v3/v4 按原布局推理。30 维权重不会自动转成 27 维。

## 初始化规则

至少十个不同观测时，使用确定性 farthest-point 覆盖采样选十点；不足时，使用观测位置、颜色和 cell 尺度，加十个固定方向的有界不同初始位置。稀疏位置扰动半径为 `0.025h`，生成位置约束在对应 cell 内；没有调用 RNG。

这些补充位置是**依据局部观测的初始化假设，不是十个已测得的真实表面点**。本轮没有法向估计、深度面片拟合或额外几何损失。

尺度来自局部不同点距离的 0.1 倍；无不同近邻时用 `0.05h`，初始化尺度限制在 `[0.001h,0.25h]`。SH DC 来自来源颜色，其余 SH 为零，Rotation 为 identity。每点初始 opacity 为 `1-0.5^(1/10)`，避免对一个近重叠十点组直接重复使用旧的 0.5。该公式仅是十点完全重叠时的组合透射率解释；不同位置、尺度和多层共存时并不保证与旧地图渲染完全等价。

## Full 梯度增长与 pruning

只使用现有 Full Render 的可见点屏幕梯度与可见次数平均。LoD、TF、Change Distill 不向 growing accumulator 写入。append 保留旧点的 accumulator/denom/radii，新行追加零；在现有 densification cadence 消费并清零窗口。

`τ0` 在第一个有正有限 L0 梯度的 Full 窗口按 p90 锁定；`α=2`。阈值完全来自训练，未读取 non-Bank 或最终质量。日志记录 Full 分辨率、金字塔级别、窗口可见统计及锁定阈值。没有把官方 `0.0002` 或旧配置 `0.001` 直接当作新阈值。

当前 rasterizer 的屏幕梯度含 `W/2,H/2` 链式因子，阈值对应的是本地这一定义。Full 仍可能使用已有金字塔分辨率，没有新增归一化或改变 Full render 调度。

- `τ_l≤g<τ_(l+1)`：按实际 Gaussian XYZ 产生同层候选。
- `g≥τ_(l+1)`：产生下一层候选。
- 仅对候选目标层去重；粗层占用不阻止细层。
- L2 的下一层请求只记受限数量，不继续加层或自动改成其它增长。
- 一个窗口使用增长前快照，新增 Anchor 不在同一窗口连锁增长。
- 初始/观测新增 Feature 置零；同层增长逐通道 max；细层增长置零。同一新 key 同时收到同层与细层提案时，以细层零初始化优先，与官方从粗到细插入的先后关系一致。

新模式直接拒绝 Gaussian clone/split；prune 把原逐点 predicate 归约为“十点全部失败”，再整组压缩属性、Feature、状态与元数据。因此组内可能保留少数无效或过大的成员，这是已批准规则的代价；没有偷偷补点或改掉原 predicate。

## Adam、训练与几何生命周期

新增 Feature 的两套 Adam owner 都保留旧状态，新行一阶/二阶 moments 为零；已有 Feature、归属和 Gaussian 不受增长扰动。张量级 Adam step 沿用既有时钟，不声称每个新行都有独立的 step=0。Gaussian 六组扩展继续保留旧 moments 并给新行追加零。

LoD 保持 joint Gaussian/Selector/Feature/TF 更新；Full 保持 Master Gaussian 更新。现有 ratio budget loss、photo loss 和 TF regularization 均保留，没有新 loss 或额外蒸馏。

Loop 的实际逐点 correction 不改变。对于同一个 block 内的混合 correction，参考中心按十个成员分别收到的变换结果平均运输；未修正成员使用 identity。参考尺度同样按成员占比运输。Feature、UID、十点 owner 和 moments 不因碰撞平均或合并。随后更新固定原点的空间 key 桶，已有 CD 在新布局上继续训练对应 Selector/TF。

维护在 render lock、无存活 backward 的边界执行；geometry capture 到 finalize 期间禁止 pruning。官方基于相机距离的 LoD、extra-level、神经 Gaussian 属性生成均未引入。

## 验证与质量

证据目录：`tests/anchor_selector/explicit_validation/`。隔离服务器目录：`/home/prmi/zwj/octree-explicit-20261021/`，没有覆盖既有生产工程。

功能 gate 包括：CPU/CUDA 固定十点和增长、native/Python 四 ratio 一致性、旧 27 维回归、真实 Mapping 的 joint/Full 梯度分流、Full-only accumulator、TF、global/local Change Distill、append/prune 和 checkpoint。几何单元另含同层 key 碰撞、混合 correction 和 UID 保留。

质量协议为 `fr1_xyz`，seed 2026/2027，27 维固定 Anchor 与新十点多层模式分别重新训练 10,000 Mapping。先通过功能 gate，再锁定共同 GT 视图与共同绝对 K，使用最终 checkpoint，保留两个 seed。Primary 为生产 Cached Bernoulli，secondary 为概率排序的相同 K 诊断，不改变采样器或导出语义。

**结论：首轮支持新的地图表示改善 LoD；Full-map 未全面通过回归。**生产 Cached Bernoulli 与 same-K 各 8/8 个对照均提升 PSNR/SSIM、降低 LPIPS。same-K PSNR 增益为 0.578–1.553 dB，各 case 的 PSNR 改善视图占比为 84.7%–100%。这同时改变了初始几何、opacity、pruning、地图容量和层级增长，不能将收益单独归因于层级。

### Full-map

| Seed | 方法 | Gaussian | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|---:|
| 2026 | 固定 Anchor | 25989 | 22.2344 | 0.77415 | 0.24547 |
| 2026 | 多层十点 | 28760 | 21.9894 | 0.76055 | 0.29102 |
| 2027 | 固定 Anchor | 25675 | 22.0000 | 0.76404 | 0.25095 |
| 2027 | 多层十点 | 30190 | 22.1007 | 0.76340 | 0.27901 |

新−旧 Full PSNR 为 −0.245/+0.101 dB；SSIM 为 −0.01360/−0.00064；LPIPS 为 +0.04555/+0.02806。LPIPS 越低越好，此项两个 seed 均回退。

### LoD 生产 Cached Bernoulli

| Seed | Ratio | 实际 N：固定 / 新 | PSNR：固定 / 新 | ΔSSIM | ΔLPIPS |
|---|---|---:|---:|---:|---:|
| 2026 | 1% | 261 / 305 | 16.6090 / 18.4898 | +0.05252 | -0.08241 |
| 2026 | 5% | 1359 / 1434 | 18.8999 / 20.5549 | +0.04313 | -0.08087 |
| 2026 | 10% | 2690 / 2868 | 20.0561 / 21.0315 | +0.02439 | -0.03701 |
| 2026 | 15% | 3888 / 4363 | 20.3011 / 21.1124 | +0.01927 | -0.02924 |
| 2027 | 1% | 264 / 297 | 17.0795 / 18.5192 | +0.07951 | -0.04875 |
| 2027 | 5% | 1346 / 1485 | 18.6848 / 19.9553 | +0.07572 | -0.05560 |
| 2027 | 10% | 2666 / 3035 | 19.4267 / 20.1501 | +0.05683 | -0.02013 |
| 2027 | 15% | 3910 / 4617 | 19.6029 / 20.1627 | +0.05104 | -0.00664 |

### LoD 相同 K

K 在查看质量前锁定为 256/1283/2567/3851。此处只作预算控制诊断，训练和生产导出继续 Cached Bernoulli。

| Seed | Ratio | K | PSNR：固定 / 新 | ΔSSIM | ΔLPIPS |
|---|---|---:|---:|---:|---:|
| 2026 | 1% | 256 | 16.5693 / 18.1168 | +0.04744 | -0.07549 |
| 2026 | 5% | 1283 | 18.8143 / 20.3678 | +0.04106 | -0.07499 |
| 2026 | 10% | 2567 | 19.8883 / 21.0097 | +0.02754 | -0.03918 |
| 2026 | 15% | 3851 | 20.2519 / 21.0880 | +0.01983 | -0.02938 |
| 2027 | 1% | 256 | 16.9595 / 17.7128 | +0.07017 | -0.03249 |
| 2027 | 5% | 1283 | 18.5525 / 19.6475 | +0.06702 | -0.04308 |
| 2027 | 10% | 2567 | 19.3881 / 20.1289 | +0.05658 | -0.01681 |
| 2027 | 15% | 3851 | 19.5711 / 20.1492 | +0.05170 | -0.00722 |

绝对 SSIM/LPIPS、逐视图结果、paired mean/median/p10/p90/改善比例完整保存于 `quality_summary.csv`、`quality_per_view.csv`、`paired_statistics.csv`。

### 层级与增长

| Seed | L0 / L1 / L2 Anchor | 最终 Anchor | 总创建 / 整组 prune | 同层 / 细层候选 | L2 受限请求 |
|---|---:|---:|---:|---:|---:|
| 2026 | 821 / 1763 / 292 | 2876 | 11919 / 9043 | 733 / 2577 | 99 |
| 2027 | 851 / 1824 / 344 | 3019 | 12000 / 8981 | 804 / 2786 | 36 |

累计创建含初始化、在线观测与梯度 growing，也包含已 prune 区域被新观测重新创建；候选数不等于去重后的新 Anchor 数。两个模型最终每 Anchor 均 exact 10 点、UID 唯一。每次均有 19 个增长窗口；锁定 τ0 分别为 0.0006614705/0.0006414179。统计日志记录到的 Full 分辨率为 640×480，未因新方法改掉训练分辨率。高累计 prune 数提示存在明显拓扑周转，本轮只记录，不改规则。

### 时间与显存

| Seed | 方法 | 训练 wall 秒 | Full / LoD 次数 | Native Selector ms | GPU peak allocated MiB |
|---|---|---:|---:|---:|---:|
| 2026 | fixed | 68.967 | 4748 / 8688 | 0.2133 | 546.919 |
| 2026 | explicit | 66.099 | 4607 / 8652 | 0.2501 | 549.934 |
| 2027 | fixed | 67.987 | 4660 / 8666 | 0.2287 | 557.212 |
| 2027 | explicit | 65.389 | 4598 / 8650 | 0.2632 | 537.688 |

Native Selector 是实际 population 的 LibTorch forward，同一测量协议、20 次预热、5×200 calls；新方法分别慢 17.3%/15.1%，包含总点数增加，不能当作相同 N 下的纯结构开销。Anchor Feature 加一套两份 Adam moments 的理论占用从 0.236/0.259 MiB 增至 0.527/0.553 MiB；两套 optimizer 都已建立 moments 时还要另加一套 moments。新布局 buffer 为 0.376/0.394 MiB。

新模式已记录到的 insert 维护累计为 153.0/150.0 ms，调用 p95 为 2.419/2.128 ms；此项包含初始化/插入、没有覆盖全部 prune、候选扫描与几何维护成本，不能作为总维护耗时。训练 wall 含异步 SLAM，且 Full/LoD 次数不同，不能据 66.10/65.39 秒低于 baseline 直接宣称同轨迹加速。

同步 rasterizer 计时：Full 固定→新为 0.6757→0.5252 ms、0.6557→0.5174 ms；same-K LoD 固定为 0.4955–0.5680 ms，新为 0.4725–0.5222 ms。该指标不含预先缓存的 Selector、TF 和导出准备，几何/屏幕覆盖也不同，不能等同端到端 LoD latency。逐 case 时间与 p90 见 `render_runtime.csv`。

### 功能门槛与证据边界

| 验证项 | 状态 |
|---|---|
| CPU/CUDA 十点、同层/细层增长、L2 上限、Feature、两套 Adam、整组 pruning、UID/碰撞/混合 correction、保存加载 | TESTED |
| CPU/CUDA 四 ratio native/Python 推理及旧 27 维回归 | TESTED |
| 实际 Renderer：joint LoD、Master-only Full、Full-only 梯度统计、TF、global/local CD、append/prune | TESTED |
| 四次在线 SLAM 训练、最终十点/层级/UID/无 SI-GI、共同视图和 same-K 质量 | TESTED |
| 真实数据中的 Loop closure 碰撞迁移 | PARTIALLY VERIFIED：本场景两次在线运行均无 LOCAL/GLOBAL_GEOMETRY 事件；已通过生产 Mapping 功能路径与合成碰撞验证 |
| 跨场景、容量匹配因果对照、完整训练 resume | NOT IMPLEMENTED / NOT TESTED |

基于当前证据，LoD 方向值得保留，但 Full 回退需要单独研究；本轮不自动增加层级、调阈值、改初始化、改 Loss 或追加新算法。

## 可比性与限制

新增地图表示改变了初始容量、增容和 pruning；在线 Full/LoD 混合及 SLAM 几何轨迹也可能不同。same-K 仅控制评价时选中点数，不能消除完整地图容量与在线训练轨迹差异，质量变化不能直接归因于层级组织。

旧任意 occupancy checkpoint 不能直接成为合法的十点地图，本轮从 SLAM 观测重新训练。现有 PLY/Selector 导出仍为推理快照，不是包含所有 optimizer/在线调度状态的训练恢复格式；本轮没有实现跨旧模型的自动转换或完整训练 resume。

本轮没有 stage、commit 或 push。
