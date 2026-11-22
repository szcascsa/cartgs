# Octree-inspired 多层 Anchor 显式 Gaussian：源码审查与实施方案

日期：2026-10-20。基线：`03e135b9083e49bd868ac776a9fa221504e2bb7d`，分支 `redundancy-aware-si`。

**决策更新：用户已确认 C1-A、C2-A、C3-A；允许整组 pruning、有界初始化和稳定 UID 碰撞桶。正在按此实施。本文件下文保留实施前审查记录，实际实现和验证状态以最终重构报告为准。**

## 1. 当前代码实际做了什么

| 环节 | 源码位置 | 当前真实实现 | 本轮需要改变的部分 |
|---|---|---|---|
| 地图初始化 | `src/gaussian_model.cpp:111`；`src/gaussian_mapper.cpp:1116,1281` | 每个输入观测点直接成为一个显式 Gaussian，随后通过 reset 回调构建 Selector 网格 | 先构建多层 Anchor，再为每个 Anchor 一次创建 10 个显式 Gaussian |
| 在线观测 | `src/gaussian_model.cpp:202,294`；`src/gaussian_mapper.cpp:3470,3569,4046` | 新地图点、深度缓存按点 append；Local BA/Loop 分支还要求一批至少 10 个输入点 | 改为按目标层级的空间 key 去重，以完整 Anchor block 追加；不能因输入不足旧批大小永久丢失待处理观测 |
| Anchor 归属 | `src/gaussian_mapper.cpp:142`；`improvements/selector/anchor_bank.h:69` | Gaussian 先增加，Selector 随后按当前位置分配 Feature；Anchor 不负责地图容量 | Anchor 持有固定 10 个 slot，追加与删除必须是两者的原子操作 |
| 当前八叉细化 | `improvements/selector/anchor_bank.h:170` | occupancy>128 时把父节点里的既有 Gaussian 路由到非空子节点，最多 L2；父节点成为历史模板 | 彻底停用此新路径中的 occupancy refine；新增细层 Anchor 时父层继续持有自己的 10 点 |
| Gaussian 增容 | `src/gaussian_model.cpp:872` | 梯度触发 clone 和 split；split 删除原 Gaussian | 新路径禁止调用 clone/split，只允许创建完整 Anchor block |
| Gaussian 删除 | `src/gaussian_model.cpp:611,881` | opacity、屏幕半径、世界尺度生成逐点删除 mask | 需要明确整 Anchor pruning 规则，见 C1 |
| 增长统计 | `src/gaussian_mapper.cpp:1696`；`src/gaussian_model.cpp:895` | 已经只累计 Full Render 的屏幕位置梯度，LoD 不累计 | 可复用统计来源，但需修正 append 时清空全图累计量的行为 |
| Selector | `improvements/selector/gumbel_network.h:33,201,221` | 同时支持 20/27/30 维；27 维为 27→32 ReLU→32 ReLU→2 | 本轮明确使用 27 维，关闭 Global XYZ；保留原网络及 Cached Bernoulli |
| 几何事件 | `src/gaussian_model.cpp:427`；`src/gaussian_mapper.cpp:2513,3532` | Loop 按每点可见性和创建时间选择变换；随后按 XYZ 重新关联 Anchor，碰撞时平均 Feature/Adam moments | 不可直接沿用，否则固定 10 点归属及独立 Feature 会被破坏，见 C3 |
| 蒸馏 | `src/change_distill.cpp:252` | 已有 Selector teacher-student，扩展到受影响的旧/新 Anchor 全体成员 | 保留 loss、ratio 和 TF 迁移，适配新 block 身份及 Feature 布局 |
| 优化器 | `src/gaussian_mapper.cpp:2179`；`improvements/selector/anchor_bank.h:346` | Selector 参数既出现在 joint Gaussian optimizer，也出现在独立 Selector optimizer | Feature 扩展必须更新两套 owner 的参数引用和状态，不能只修改 Tensor.data |

当前固定 Anchor 的实际模型也不满足新表示。例如已有 `fixed_seed2026`：25,248 Gaussian、1,321 Feature，cell occupancy 中位数为 4，最大 472。它是上一轮最终优化地图的统计，**不是 SLAM 初始点云统计**。不能把现有 checkpoint 不加转换地当作新固定 10 点模型。

## 2. 官方参考代码核对

已下载官方 `city-super/Octree-GS` 的以下文件到项目外 `../reference/Octree-GS-20261020/`，固定 commit：

`9db64fe8981f5d6a1002c41cbc275a14f4d5ea0c`。

文件与 SHA 见该目录 `SOURCE.json`。审查覆盖 `scene/gaussian_model.py`、`train.py`、`arguments/__init__.py`、`gaussian_renderer/__init__.py`。

- `gaussian_model.py:284`：按多个 level 体素化；粗细 Anchor 共存。
- `gaussian_model.py:334`：offset slot 和初始 Anchor Feature 都为零；`arguments/__init__.py:50` 默认 10 个 slot。
- `gaussian_model.py:634`：累计可见且实际参与渲染的 offset 屏幕梯度。
- `gaussian_model.py:717`：按当前层阈值与下一层阈值区分同层/细层 growing，并按目标层去重。
- `gaussian_model.py:796`：同层新 Feature 逐通道 max；细层 Feature 为零。
- `gaussian_model.py:852`：按统计量做 Anchor growing 与整 Anchor pruning。

**不能原样复制的部分：**官方 10 个 offset slot 通过神经网络产生属性；当前任务要求 10 个真实存储、独立优化的显式 Gaussian。官方基于相机距离的层级激活、可见性 weed-out、extra-level/progressive 机制，以及从场景范围推导网格原点/尺度的方式，均不进入本轮新路径。

## 3. 需要用户确认的核心冲突

### C1：逐点 pruning 与固定 10 点结构冲突

1. **位置：**`src/gaussian_model.cpp:611,872`。
2. **当前实现：**任意 Gaussian 满足 opacity/size 删除条件便单独删除，六组属性与 Adam 同步压缩。
3. **冲突：**一个 Anchor 删除 1 点后只剩 9 点；简单把 mask 扩展为“任一点失败则整组删除”又会连带删除其余有效 Gaussian。
4. **不能直接适配的原因：**现有 mask 是单点有效性，不能无损推出整组有效性；补回单点也违反“地图增容只能创建完整新 Anchor”。
5. **方案：**A，只有 10 个成员全部满足原逐点删除 predicate 时，才删除整个 Anchor；B，新路径暂不 pruning；C，任一成员失败则删整组，不推荐。
6. **影响：**A 保留有用成员，但也会保留组内低 opacity/过大点，显存和渲染开销高于原逐点 pruning；B 容量只增不减；C 更易损伤 Full-map。没有实现或测量这些影响。
7. **推荐：**A。不补 slot，不移动其他 Anchor 的 Gaussian；整组压缩所有属性、累计量、Cached Uniform、Feature 与两套 Adam。
8. **确认：****需要。**这改变 pruning 的算法语义；不得当作普通索引修复直接实施。

### C2：稀疏观测无法直接提供 10 个不同的真实观测位置

1. **位置：**`src/gaussian_model.cpp:111,202,294`；`include/point3d.h:25`。
2. **当前实现：**输入主要只有 XYZ/RGB；没有对每个空间 cell 保证至少 10 个不同观测，也没有普遍可用的法向/局部面片。现有 opacity 初始化为 0.5，尺度来自输入点的近邻距离。
3. **冲突：**只有 1 个观测的 occupied cell 无法直接选出 10 个不同真实点。完全复制 10 次既被任务禁止，也会叠加 opacity；对复制/扰动点直接复用近邻距离估尺度还可能产生过小尺度。
4. **不能直接适配的原因：**缺少观测是实际信息不足；官方零 offset 不会替显式表示补足真实几何。
5. **方案：**A，至少 10 个不同观测时做确定性的局部覆盖采样；不足时用真实观测位置/颜色和局部尺度作确定性、有界的不同初始位置，明确标记为几何初始化假设。B，暂存不足 10 点的 cell，观测足够后再创建完整 Anchor。
6. **影响：**A 满足 occupied cell 立即拥有 10 点，但引入未直接观测的初始位置，可能影响表面和 opacity，必须单独检查 Full 初始质量；B 不制造几何，却延迟稀疏区域建图，偏离“每个 occupied cell 初始化 Anchor”。
7. **推荐：**若保留即时创建要求，采用 A；不足 10 点的扰动范围、尺度、opacity 初始化应在功能实验前写入协议。不能声称这些位置都是真实观测；不得用评价质量挑初始化。现有 0.5 opacity 不能未经检查直接重复套给十个重叠点。
8. **确认：****需要。**A/B 是对稀疏区域如何建图的不同选择，不是网络实现细节。

### C3：Loop/几何迁移、唯一空间 key、固定 10 点三者不能靠重新体素化同时保证

1. **位置：**`src/gaussian_model.cpp:427`；`src/gaussian_mapper.cpp:2513,3474,3532`；`improvements/selector/anchor_bank.h:131`。
2. **当前实现：**同一 Anchor 的成员可能分别属于不同 correction group；重新量化按当前位置换归属，Feature 与 moments 用成员平均迁移。
3. **冲突：**成员级重关联会产生任意 occupancy。即使强制整组使用同一变换，两个原本不同的 Anchor 也可能修正到同一个 `(level,key)`，合并变 20 点，保留两组则空间 key 不再唯一。固定世界原点也意味着变换后不能随意移动整个量化坐标系来掩盖碰撞。
4. **不能直接适配的原因：**一般局部修正不是对所有 block 使用同一个刚体变换；量化映射也不是单射。Change Distill 可以迁移预测，不能把这项结构矛盾消除。
5. **方案：**A，保留稳定 block UID、10 点永久归属和现有逐点几何修正，Anchor 参考位置/尺度按明确规则运输；另建固定原点的空间查询桶，允许几何事件后一个 `(level,key)` 对应多个 block，新增观测对非空桶仍去重。B，强制恢复每 key 唯一，碰撞后重采样/合并为一个 10 点 block，或扩展为整块统一修正。
6. **影响：**A 保留 Gaussian 数量、已有优化状态和 SLAM 实际逐点修正，但必须放宽“空间身份仅由 level/key 唯一确定”，且需定义混合 correction 的 Anchor 参考运输。B 可能删除/重建已有 Gaussian、改变归属或改写 SLAM 修正范围，影响质量、状态迁移与 CD 的旧点对应关系；成本更高。
7. **推荐：**A 更能保留现有 SLAM 地图与 joint 语义，但它是**显式设计变更请求**，当前没有授权把 UID 自动加进核心身份。两方案都不是完全满足原文全部条件的无损修补。若 `(level,key)` 严格唯一不可放宽，必须进一步确定碰撞时允许如何处置已有的 20 个显式 Gaussian。
8. **确认：****需要；未解决前停止生产迁移实现。**不能静默合并 Feature、丢点、挪动 Anchor 到空 cell，或仅处理测试中无碰撞的情况。

## 4. 可确定的实施结构

以下是实施方案，**尚未实现**；C1–C3 的选择需先锁定。

### 4.1 多层配置与容量管理

- 建议最大 L2，共 L0/L1/L2，`h={0.1,0.05,0.025}`，固定世界原点 `(0,0,0)`，key 使用 FP64 除法后的 `floor(x/h_l)`，包含负坐标和边界测试。
- 建议初始预定层级为 L0/L1，在线新观测也向这两层按 key 插入；L2 留给梯度触发的下一层增长。这已是多尺度初始化，并保留可测的细层增长空间。该层级协议必须在实现前最终锁定。
- 如果改为初始化和新观测直接覆盖 L0/L1/L2，同样能实现，但已观测区域的 L2 候选最初大多会被去重，不能把“无新细层 Anchor”直接当作增长代码失败；初始容量也明显更高。
- 新拓扑所有者放在地图侧，持有 level、key、reference center/scale、固定 10 slot 的归属关系；Selector 消费这些关系。不要让 Selector append/reset 回调再次自行按 Gaussian XYZ 建网格。
- 显式属性仍为 Renderer 可用的 `[10M,...]` 扁平 Tensor；正常保存次序可按 block 连续存储，同时保存 source row→Anchor/slot，compact LoD 仅创建视图索引，不改变完整地图归属。
- 同/细层 growing 仅查询目标层索引；已有粗层不能阻止细层。创建子层不删除父层，不迁移父层 10 点。
- 旧 occupancy>128 refine 与 Gaussian clone/split 在新路径禁用；保留旧方法作为 baseline。传统 append/prune API 在新模式增加整组校验，防止旁路破坏结构。

### 4.2 初始化与新观测

统一处理初始 SLAM 点、两种 `increasePcd()`、RGB-D 深度缓存；先更新观测记录，再按层查询缺失 Anchor，最后原子追加 `10×new_anchor_count` 个 Gaussian。

已存在 key 只跳过 slot 初始化。`handleNewKeyframe()`、图像/深度缓存、关键帧使用次数和现有 reconstruction optimization 继续执行，不能把去重的提前返回放到关键帧/观测处理之前。

Feature：初始为零；在线纯观测创建的新 Anchor 无候选来源时也置零。同层梯度增长按所有有效来源 Feature 逐通道 max 聚合，包括负数，不能以零为 max 的默认填充值；下一层增长置零。

不得将旧任意 occupancy checkpoint 简单 reshape 为 10 slot。建议新模型从 SLAM 观测重新初始化；旧模型仅作质量对照。网络仍为 27→32→32→2，30 维权重不能直接加载为 27 维。

### 4.3 Full 梯度统计与 growing

现有来源可复用，但需要明确以下修正：

1. `addDensificationStats()` 的 `norm(grad[:,0:2])` 已是任务要求的按可见次数平均形式。只在 Full Render 的实际可见行上累计；LoD、TF、CD 不向 growing 统计写入。
2. `cuda_rasterizer/backward.cu:550,720` 对 x/y 梯度分别乘 `W/2,H/2`。因此返回梯度具有 NDC 投影尺度的链式因子，不能称为未经尺度转换的像素坐标梯度。
3. Full 本身可能使用金字塔分辨率（`src/gaussian_mapper.cpp:1479`）。阈值预检需记录 W/H、训练 level、梯度分位数及可见次数；不自动改图像分辨率、loss 或 gradient normalization。
4. 当前 `densificationPostfix():773` 每次 append 都重置所有累计量。新路径应保留旧行，给新 10 点追加零；只在约定 growing 窗口消费/重置累计量，避免连续新观测把统计清空。
5. 复用现有 Full 安全边界及 densify cadence；该时间窗是 Mapping/Full 调度下实际收到的统计，不误称每个 Gaussian 收到了 500 次 Full。mode1 目前在线轮转 Full，tail 则 Full 后接单个轮转 LoD，不能改变此时钟。
6. 第一批候选使用预增长快照，不能在同一次 growing 中让新生 Anchor 继续连锁增长；`denom=0` 无候选，非 finite 记录并阻止该候选。
7. `tau_l=tau0*alpha^l`，`alpha>1`；处于相应区间的第 l 层 Gaussian 在其**当前实际 XYZ**产生同层/下一层候选，并按目标 level/key 去重。L2 的超阈值细化需求只计入受限统计，不降级成任意其他规则。
8. `tau0/alpha` 尚未定值。先用 Bank/训练图像收集一个预定 Full 窗口的真实统计，锁定阈值与可见次数规则后才开始正式对照；不使用 non-Bank/Full 最终质量选择阈值，不照搬官方 `0.0002` 或现有 `0.001`。

### 4.4 Feature 与 Adam

已有 Gaussian 和 Feature 数值在新增 Anchor 时保持 exact；旧归属不变。Feature append 用来源聚合得到新参数行，**两套 Adam 的新行 moments 均为零**，不能沿用上一轮“复制父 Feature 及父 moments”的 split 实现。

保持张量级 Adam step 的现有约定，不能声称新行具有独立 step=0；若要每行独立 bias-correction 时钟将是额外优化器语义变更，本方案不引入。

六组 Gaussian 参数扩展复用现有“保留旧 moments、扩展新零 moments”的逻辑。全操作在 render lock、无存活 backward graph、无正在执行 CD 的边界完成，更新 autograd leaf 和所有参数别名。group 0–5 的属性含义不变。

### 4.5 Selector、TF、CD 与导出

- 本轮按用户最新要求使用 27 维：Feature16 + 当前 Anchor 局部 Offset3 + activated Scale3 + SH DC3 + activated Opacity1 + logRatio1；不含 Global XYZ、camera、level 或 occupancy 额外输入。
- 当前代码将 cell-local offset 的启用与 Global XYZ 开关绑定，27 维默认用 root offset。需要解除此耦合；按 checkpoint 的布局元数据确定 offset 语义，不能为了获得 cell-local offset 又打开 30 维输入。
- Offset 只由真实 XYZ、Anchor 参考位置和 `s_a` 计算，不创建独立可漂移 offset 参数。常规 Gaussian 梯度更新不改变其 slot owner。
- joint LoD 更新 Gaussian、Selector、Feature、就绪 TF；Full 只更新 Gaussian。现有 loss 中 ratio budget 项及 TF regularization 原样保留，既不新增，也不误删为“只有 photo”。SI/GI 新模式显式禁止，避免继承历史 ablation 配置意外启用 teacher。
- 仍使用 Cached Bernoulli 训练与生产导出；same-K probability ranking 只作预算控制诊断。实际选中数量可不等于 ratio×N，不将其报告为 Exact-K。
- CD 使用已有 teacher 和现有损失，扩展新布局复制与受影响 block 范围；不在 capture 到 CD 完成期间 prune/reorder。C3 决定之前，不能定义正确的混合修正 block 迁移。
- 保存 Anchor level/key/reference、Feature、10 slot 映射、实际 Gaussian 数、布局版本、Cached Uniform、TF 关联元数据。Python/native 都校验 `N=10M` 和每 owner exact 10 点，不靠输入宽度 27 猜测旧/新 offset 语义。

## 5. 验证顺序及质量协议

### 功能门槛

1. CPU 小模型：初始化每 Anchor exact 10 点，属性不完全重复，空/稀疏观测规则与协议一致，负坐标/边界 key 稳定。
2. 重复在线观测不追加同层同 key；同位置不同 level 可共存；加入已占区域的新图像仍参与优化。
3. 同层/细层增长分别用可控梯度触发；去重、最大 level、父子共存、旧 block 不迁移均检查；新点完整按 10 追加。
4. Feature max 聚合含全负数例；细层/初始零 Feature；旧 Feature/属性和 Adam exact，新行两套 moments 为零；实际 backward 后再次 append/prune。
5. clone/split 不可执行；prune 只能删除完整 block；load、导出、compact selection 不破坏存储。
6. 每个 ratio 的 Full/LoD 梯度分流、Cached mask/native-Python 一致、view-independent/SI-GI disabled、TF 与既有 CD 回归。
7. 几何测试包含一个 block 被多 correction 命中、两个 block 修正后同 key、scale refinement、追加观测与 CD 同事件，不能只验证无碰撞情况。

### 在线质量对照

功能通过后，建议复用 `fr1_xyz`、2026/2027、每 arm 10,000 Mapping、相同 ratio/Full 调度和共同评价视图作为首轮。Baseline 使用 27 维固定 Anchor；Candidate 使用本轮多层固定 10 点新表示，SI/GI 均关闭。其它 scene 需在首轮通过后按事先锁定配置扩展，不自动搜索参数。

必须同时报告 Full PSNR/SSIM/LPIPS、Full Gaussian/Anchor 数与各层分布、实际 LoD 数量、same-K LoD PSNR/SSIM/LPIPS/photo、训练耗时、Selector/映射维护/增长耗时及峰值显存。共同 K 在训练结束、查看质量前按所有模型可用总数锁定，不用单个最好 seed。

**same-K LoD 仅控制评价时选中点数，不能抵消不同完整地图容量和在线建图轨迹的影响。**即使 Candidate 更好，也不能直接归因于层级组织；若需要证明这一点，后续要单独设计容量匹配对照，不在本轮暗中加入总量上限。

显存静态估算：SH degree=3 时，每 Gaussian 有 59 个 float 属性，纯属性约 236 B；参数加两份 Adam moments 约 708 B，不含梯度、rasterizer、统计和 Tensor 开销。每新 Anchor 至少多 10 组属性和 1 个 Feature；真实显存/速度必须运行测量，不能用此估算宣称性能已达标。

## 6. 当前交付与阻塞

| 项目 | 状态 |
|---|---|
| 本地源码调用链、官方参考与固定版本核对 | REVIEWED |
| 数据结构、接入点、功能/质量验证协议 | PROPOSED |
| C1 pruning、C2 稀疏初始化、C3 几何碰撞决策 | APPROVED：用户已选择全部推荐方案 |
| 地图初始化、固定 10 点和梯度 growing 生产实现 | IMPLEMENTED / TESTED，见最终报告 |
| 新模型 correctness、在线训练、质量与性能 | 9 gates、4 models、36 evaluations 完成，见最终报告 |
| Git stage / commit / push | NOT PERFORMED |

停止修改相关生产链是用户任务明确要求：涉及固定 10 点、初始化、pruning、增长或训练语义的冲突，必须先报告，不能自行改变核心设计。此处没有自动扩大网络、改 loss、恢复 SI/GI，或把上一轮通过的测试当作本轮新结构已验证。
